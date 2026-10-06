"""
Training script for Module 3 — AutoVC Voice Conversion

Trains the AutoVC autoencoder on the 30 LibriSpeech dev-clean *train* speakers with
the losses from the AutoVC paper:
  - Self-reconstruction: with target = source speaker, decoder (+ postnet) output
    must reconstruct the input mel
  - Content-code consistency: re-encoding the reconstruction gives the same codes

Speaker encoder is Module 2's trained, frozen ECAPA-TDNN — not retrained here.
Each speaker is represented by the mean of its (L2-normalized) utterance embeddings,
as in AutoVC.

Outputs:
  - checkpoints/voice_conversion.pt
  - results/loss_curves/voice_conversion_loss.png
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
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.voice_conversion import (audio_to_mel, build_model, content_consistency_loss,
                                     self_reconstruction_loss, speaker_embedding)
from training.data_utils import load_audio, load_manifest, set_seed, speakers_in, utterances


def peak_normalize(audio):
    return audio / (np.abs(audio).max() + 1e-8) * 0.9


def precompute(manifest, speakers, emb_utts_per_speaker=20):
    """Log-mels for every utterance + one mean embedding per speaker."""
    mels = {'train': [], 'val': []}
    spk_emb = {}
    for spk in tqdm(speakers, desc="Precomputing mels + speaker embeddings"):
        embs = []
        for subset in ('train', 'val'):
            for i, path in enumerate(utterances(manifest, spk, subset)):
                audio = peak_normalize(load_audio(path))
                mels[subset].append((audio_to_mel(audio), spk))
                if subset == 'train' and i < emb_utts_per_speaker:
                    embs.append(speaker_embedding(audio))
        mean = np.mean(embs, axis=0)
        spk_emb[spk] = (mean / np.linalg.norm(mean)).astype(np.float32)
    return mels, spk_emb


def sample_batch(items, spk_emb, batch_size, crop, rng):
    mels, embs = [], []
    for mel, spk in rng.sample(items, batch_size):
        if mel.shape[1] > crop:
            start = rng.randint(0, mel.shape[1] - crop)
            mel = mel[:, start:start + crop]
        else:
            mel = np.pad(mel, ((0, 0), (0, crop - mel.shape[1])), constant_values=np.log(1e-5))
        mels.append(mel)
        embs.append(spk_emb[spk])
    return torch.from_numpy(np.stack(mels)), torch.from_numpy(np.stack(embs))


def train(config_path='configs/voice_conversion.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    train_cfg, loss_cfg = config['training'], config['loss']
    set_seed(train_cfg.get('seed', 42))
    rng = random.Random(train_cfg.get('seed', 42))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    manifest = load_manifest()
    speakers = speakers_in(manifest, 'train')
    mels, spk_emb = precompute(manifest, speakers)
    print(f"{len(speakers)} speakers | {len(mels['train'])} train / {len(mels['val'])} val utterances")

    model = build_model(config).to(device)
    # Per-bin normalization statistics from the training mels
    all_frames = np.concatenate([m for m, _ in mels['train']], axis=1)
    model.mel_mean.copy_(torch.from_numpy(all_frames.mean(1, keepdims=True)))
    model.mel_std.copy_(torch.from_numpy(all_frames.std(1, keepdims=True) + 1e-5))
    del all_frames
    print(f"AutoVC parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.get('learning_rate', 1e-3))
    total_steps = train_cfg.get('steps', 8000)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps, eta_min=1e-5)
    batch_size = train_cfg.get('batch_size', 16)
    crop = train_cfg.get('crop_frames', 128)
    val_every = train_cfg.get('val_every', 500)
    w_recon = loss_cfg.get('reconstruction_weight', 1.0)
    w_content = loss_cfg.get('content_consistency_weight', 1.0)

    # Fixed validation batch (held-out utterances of the same speakers)
    val_rng = random.Random(0)
    val_batches = [sample_batch(mels['val'], spk_emb, min(16, len(mels['val'])), crop, val_rng)
                   for _ in range(8)]

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results/loss_curves', exist_ok=True)
    hist = {'step': [], 'train': [], 'val': []}
    best_val, running, start = float('inf'), [], time.time()

    model.train()
    for step in range(1, total_steps + 1):
        mel, emb = sample_batch(mels['train'], spk_emb, batch_size, crop, rng)
        mel, emb = model.normalize(mel.to(device)), emb.to(device)

        mel_out, mel_post, codes = model(mel, emb, emb)
        loss_recon = self_reconstruction_loss(mel_out, mel_post, mel)
        loss_content = content_consistency_loss(model, mel_post, codes, emb)
        loss = w_recon * loss_recon + w_content * loss_content

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        running.append(loss_recon.item())

        if step % val_every == 0 or step == total_steps:
            model.eval()
            with torch.no_grad():
                val = float(np.mean([
                    self_reconstruction_loss(*model(model.normalize(m.to(device)), e.to(device),
                                                    e.to(device))[:2],
                                             model.normalize(m.to(device))).item()
                    for m, e in val_batches]))
            model.train()
            hist['step'].append(step)
            hist['train'].append(float(np.mean(running)))
            hist['val'].append(val)
            running = []
            flag = ''
            if val < best_val:
                best_val = val
                torch.save({'step': step, 'model_state_dict': model.state_dict(),
                            # tensors (not numpy) so torch.load(weights_only=True) works
                            'speaker_embeddings': {s: torch.from_numpy(e) for s, e in spk_emb.items()},
                            'val_loss': val},
                           'checkpoints/voice_conversion.pt')
                flag = '  [saved best]'
            print(f"step {step:6d} | recon train {hist['train'][-1]:.4f} | val {val:.4f} | "
                  f"content {loss_content.item():.4f} | {(time.time() - start) / 60:.1f} min{flag}",
                  flush=True)

    plt.figure(figsize=(10, 6))
    plt.plot(hist['step'], hist['train'], label='Train', marker='o', markersize=3)
    plt.plot(hist['step'], hist['val'], label='Val (held-out utterances)', marker='s', markersize=3)
    plt.xlabel('Step')
    plt.ylabel('Self-reconstruction loss (MSE, normalized mel)')
    plt.title('Module 3: AutoVC Voice Conversion — Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/loss_curves/voice_conversion_loss.png', dpi=150)
    plt.close()
    print(f"\nBest val reconstruction loss {best_val:.4f}. Training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train AutoVC voice conversion model')
    parser.add_argument('--config', default='configs/voice_conversion.yaml')
    parser.add_argument('--threads', type=int, default=None, help='CPU threads for torch')
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    train(args.config)
