"""
Training script for Module 2 — ECAPA-TDNN Speaker Embeddings

Trains the ECAPA-TDNN speaker-embedding network with AAM-Softmax as a closed-set
classifier over the 30 LibriSpeech dev-clean *train* speakers. After training the
AAM-Softmax class weights are discarded; the embedding is used for diarization and
as Module 3's speaker-identity encoder.

Split: per-speaker train/val utterances from data/manifest.json (fixed, no overlap).
Held-out test speakers are only used by evaluation/eval_diarization.py.

Augmentation: with probability 0.5 a different train speaker is mixed in at
-20..-10 dB, so embeddings tolerate the cross-talk left in separated streams.

Outputs:
  - checkpoints/diarization.pt
  - results/loss_curves/diarization_loss_acc.png
  - results/metrics/tsne_embeddings.png (--tsne)
"""

import argparse
import os
import random
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.diarization import SAMPLE_RATE, AAMSoftmax, build_model, extract_log_mel
from training.data_utils import (load_audio, load_manifest, random_crop, rms_normalize,
                                 set_seed, speakers_in, utterances)


class SpeakerDataset(Dataset):
    """(log-mel, speaker index) pairs from a fixed list of utterances."""

    def __init__(self, items, data_cfg, crop_sec, augment=False, interferers=None):
        self.items = items  # list of (path, label)
        self.data_cfg = data_cfg
        self.crop = int(crop_sec * SAMPLE_RATE)
        self.augment = augment
        self.interferers = interferers or []

    def __len__(self):
        return len(self.items)

    def _mel(self, audio):
        d = self.data_cfg
        return extract_log_mel(audio, SAMPLE_RATE, n_mels=d['n_mels'], n_fft=d['n_fft'],
                               hop_length=d['hop_length'], win_length=d['win_length'])

    def __getitem__(self, idx):
        path, label = self.items[idx]
        audio = load_audio(path)
        if self.augment:
            audio = random_crop(audio, self.crop)
            if random.random() < 0.5:
                other_path, other_label = random.choice(self.interferers)
                if other_label != label:
                    other = random_crop(load_audio(other_path), self.crop)
                    snr = random.uniform(10, 20)
                    audio = rms_normalize(audio) + rms_normalize(other) * 10 ** (-snr / 20)
        else:
            # Deterministic centre crop for validation
            if len(audio) > self.crop:
                start = (len(audio) - self.crop) // 2
                audio = audio[start:start + self.crop]
            else:
                audio = np.pad(audio, (0, self.crop - len(audio)))
        return torch.from_numpy(self._mel(audio)), label


def make_items(manifest, speakers, subset):
    return [(path, idx) for idx, spk in enumerate(speakers)
            for path in utterances(manifest, spk, subset)]


def train(config_path='configs/diarization.yaml', num_workers=2):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    train_cfg, loss_cfg, data_cfg = config['training'], config['loss'], config['data']
    set_seed(train_cfg.get('seed', 42))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    manifest = load_manifest()
    speakers = speakers_in(manifest, 'train')
    train_items = make_items(manifest, speakers, 'train')
    val_items = make_items(manifest, speakers, 'val')
    crop_sec = train_cfg.get('crop_sec', 2.0)

    train_loader = DataLoader(
        SpeakerDataset(train_items, data_cfg, crop_sec, augment=True, interferers=train_items),
        batch_size=train_cfg.get('batch_size', 32), shuffle=True, drop_last=True,
        num_workers=num_workers, persistent_workers=num_workers > 0)
    val_loader = DataLoader(SpeakerDataset(val_items, data_cfg, crop_sec),
                            batch_size=64, shuffle=False, num_workers=0)
    print(f"{len(speakers)} speakers | {len(train_items)} train / {len(val_items)} val utterances")

    model = build_model(config).to(device)
    aam_loss = AAMSoftmax(config['model'].get('embedding_dim', 192), len(speakers),
                          margin=loss_cfg.get('margin', 0.2), scale=loss_cfg.get('scale', 30)).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    params = list(model.parameters()) + list(aam_loss.parameters())
    optimizer = torch.optim.Adam(params, lr=train_cfg.get('learning_rate', 1e-3),
                                 weight_decay=train_cfg.get('weight_decay', 2e-5))
    epochs = train_cfg.get('epochs', 30)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=train_cfg.get('learning_rate', 1e-3),
        total_steps=epochs * len(train_loader), pct_start=0.15)

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results/loss_curves', exist_ok=True)
    hist = {k: [] for k in ('train_loss', 'val_loss', 'train_acc', 'val_acc')}
    best_val_acc, start = -1.0, time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for mel, labels in train_loader:
            mel, labels = mel.to(device), labels.to(device)
            emb = model(mel)
            loss = aam_loss(emb, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * len(labels)
            correct += (aam_loss.cosine(emb).argmax(1) == labels).sum().item()
            total += len(labels)

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for mel, labels in val_loader:
                mel, labels = mel.to(device), labels.to(device)
                emb = model(mel)
                v_loss += aam_loss(emb, labels).item() * len(labels)
                v_correct += (aam_loss.cosine(emb).argmax(1) == labels).sum().item()
                v_total += len(labels)

        hist['train_loss'].append(loss_sum / total)
        hist['train_acc'].append(100 * correct / total)
        hist['val_loss'].append(v_loss / v_total)
        hist['val_acc'].append(100 * v_correct / v_total)

        flag = ''
        if hist['val_acc'][-1] >= best_val_acc:
            best_val_acc = hist['val_acc'][-1]
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'speakers': speakers, 'val_acc': best_val_acc}, 'checkpoints/diarization.pt')
            flag = '  [saved best]'
        print(f"epoch {epoch:3d} | loss {hist['train_loss'][-1]:.3f}/{hist['val_loss'][-1]:.3f} | "
              f"acc {hist['train_acc'][-1]:5.1f}%/{hist['val_acc'][-1]:5.1f}% | "
              f"{(time.time() - start) / 60:.1f} min{flag}", flush=True)

    ep = range(1, epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(ep, hist['train_loss'], label='Train', marker='o', markersize=3)
    ax1.plot(ep, hist['val_loss'], label='Val', marker='s', markersize=3)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('AAM-Softmax Loss'); ax1.set_title('Loss')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(ep, hist['train_acc'], label='Train (augmented)', marker='o', markersize=3)
    ax2.plot(ep, hist['val_acc'], label='Val (held-out utterances)', marker='s', markersize=3)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'Closed-set speaker accuracy ({len(speakers)} speakers)')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.suptitle('Module 2: ECAPA-TDNN — Training Curves')
    plt.tight_layout()
    plt.savefig('results/loss_curves/diarization_loss_acc.png', dpi=150)
    plt.close()
    print(f"\nBest val accuracy: {best_val_acc:.1f}%. Training complete.")


def visualize_tsne(config_path='configs/diarization.yaml'):
    """t-SNE of embeddings for the 10 held-out test speakers (never seen in training)."""
    from models.diarization import get_embedding

    manifest = load_manifest()
    speakers = speakers_in(manifest, 'test')
    rng = random.Random(0)
    embs, labels = [], []
    for spk in speakers:
        for path in rng.sample(utterances(manifest, spk, 'train'), 25):
            embs.append(get_embedding(load_audio(path), SAMPLE_RATE))
            labels.append(spk)
    embs = np.stack(embs)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    labels = np.array(labels)

    pts = TSNE(n_components=2, random_state=42, perplexity=20, metric='cosine').fit_transform(embs)
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(speakers)))
    for color, spk in zip(colors, speakers):
        mask = labels == spk
        plt.scatter(pts[mask, 0], pts[mask, 1], color=color, label=f'Speaker {spk}', alpha=0.75, s=22)
    plt.title('Module 2: Embeddings of 10 unseen test speakers (t-SNE)')
    plt.xlabel('t-SNE 1'); plt.ylabel('t-SNE 2')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs('results/metrics', exist_ok=True)
    plt.savefig('results/metrics/tsne_embeddings.png', dpi=150)
    plt.close()
    print("t-SNE plot saved to results/metrics/tsne_embeddings.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ECAPA-TDNN speaker embedding model')
    parser.add_argument('--config', default='configs/diarization.yaml')
    parser.add_argument('--tsne', action='store_true', help='Generate t-SNE visualization')
    parser.add_argument('--threads', type=int, default=None, help='CPU threads for torch')
    parser.add_argument('--workers', type=int, default=2, help='DataLoader workers')
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    if args.tsne:
        visualize_tsne(args.config)
    else:
        train(args.config, num_workers=args.workers)
