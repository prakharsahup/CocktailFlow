"""
Training script for Module 1 — Conv-TasNet Speech Separation

Trains on 2-speaker mixtures of LibriSpeech dev-clean *train* speakers using SI-SDR
loss with utterance-level Permutation Invariant Training (uPIT).

Mixtures are generated on the fly (dynamic mixing): every step draws two random
train-speaker utterances, crops them, and mixes them at a random relative gain.
With only ~4 h of training audio this gives far more variety than a fixed set.

Validation uses the fixed mixtures in data/mixtures/val (train speakers, held-out
utterances); the best-validation weights are saved. Test metrics are computed
separately by evaluation/eval_separation.py on held-out speakers.

Outputs:
  - checkpoints/separation.pt
  - results/loss_curves/separation_loss.png
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
import soundfile as sf
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, IterableDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.separation import SAMPLE_RATE, build_model, pit_loss
from training.data_utils import (abs_path, load_manifest, rms_normalize,
                                 set_seed, speakers_in, utterances)


class DynamicMixDataset(IterableDataset):
    """Endless stream of (mixture, sources) built from random train-speaker pairs."""

    def __init__(self, manifest, segment_sec, seed):
        self.speakers = speakers_in(manifest, 'train')
        self.utts = {s: utterances(manifest, s, 'train') for s in self.speakers}
        self.segment = int(segment_sec * SAMPLE_RATE)
        self.seed = seed
        self._lengths = {}

    def _crop(self, path, rng):
        """Read a random `segment`-long crop straight from disk (no full-file load)."""
        if path not in self._lengths:
            self._lengths[path] = sf.info(abs_path(path)).frames
        n = self._lengths[path]
        start = rng.randint(0, n - self.segment) if n > self.segment else 0
        audio, _ = sf.read(abs_path(path), start=start, frames=self.segment, dtype='float32')
        return np.pad(audio, (0, self.segment - len(audio)))

    def __iter__(self):
        rng = random.Random(self.seed)
        while True:
            spk1, spk2 = rng.sample(self.speakers, 2)
            w1 = self._crop(rng.choice(self.utts[spk1]), rng)
            w2 = self._crop(rng.choice(self.utts[spk2]), rng)
            if np.abs(w1).max() < 1e-4 or np.abs(w2).max() < 1e-4:
                continue  # crop landed in silence
            w1 = rms_normalize(w1)
            w2 = rms_normalize(w2) * 10 ** (rng.uniform(-5, 5) / 20)
            gain = 10 ** (rng.uniform(-6, 6) / 20)  # overall level augmentation
            sources = np.stack([w1, w2]).astype(np.float32) * gain
            mix = sources.sum(0, keepdims=True)
            yield torch.from_numpy(mix), torch.from_numpy(sources)


class FixedMixDataset(Dataset):
    """Fixed mixtures written by scripts/prepare_data.py (mix_clean/, s1/, s2/)."""

    def __init__(self, split_dir):
        self.split_dir = split_dir
        self.files = sorted(os.listdir(os.path.join(split_dir, 'mix_clean')))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        read = lambda d: sf.read(os.path.join(self.split_dir, d, name), dtype='float32')[0]
        mix = torch.from_numpy(read('mix_clean')).unsqueeze(0)
        sources = torch.from_numpy(np.stack([read('s1'), read('s2')]))
        return mix, sources


@torch.no_grad()
def validate(model, dataset, device):
    model.eval()
    losses = [pit_loss(model(mix.unsqueeze(0).to(device)), src.unsqueeze(0).to(device)).item()
              for mix, src in dataset]
    model.train()
    return float(np.mean(losses))


def train(config_path='configs/separation.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    train_cfg = config['training']
    set_seed(train_cfg.get('seed', 42))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = build_model(config['model']).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    manifest = load_manifest()
    train_loader = DataLoader(
        DynamicMixDataset(manifest, train_cfg.get('segment_sec', 3.0), train_cfg.get('seed', 42)),
        batch_size=train_cfg.get('batch_size', 4), num_workers=0)
    val_dataset = FixedMixDataset(os.path.join('data', 'mixtures', 'val'))

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.get('learning_rate', 1e-3),
                                 weight_decay=train_cfg.get('weight_decay', 0.0))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    total_steps = train_cfg.get('steps', 6000)
    val_every = train_cfg.get('val_every', 500)
    grad_clip = train_cfg.get('grad_clip', 5.0)

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results/loss_curves', exist_ok=True)

    history = {'step': [], 'train': [], 'val': []}
    best_val = float('inf')
    running, start = [], time.time()
    print(f"\nTraining for {total_steps} steps (val every {val_every}), "
          f"{len(val_dataset)} val mixtures")

    model.train()
    for step, (mix, sources) in enumerate(train_loader, start=1):
        mix, sources = mix.to(device), sources.to(device)
        loss = pit_loss(model(mix), sources)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        running.append(loss.item())

        if step % val_every == 0 or step == total_steps:
            val_loss = validate(model, val_dataset, device)
            train_loss = float(np.mean(running))
            running = []
            scheduler.step(val_loss)
            history['step'].append(step)
            history['train'].append(train_loss)
            history['val'].append(val_loss)
            flag = ''
            if val_loss < best_val:
                best_val = val_loss
                torch.save({'step': step, 'model_state_dict': model.state_dict(),
                            'val_loss': val_loss}, 'checkpoints/separation.pt')
                flag = '  [saved best]'
            print(f"step {step:6d} | train {train_loss:7.3f} | val {val_loss:7.3f} | "
                  f"lr {optimizer.param_groups[0]['lr']:.1e} | "
                  f"{(time.time() - start) / 60:.1f} min{flag}", flush=True)

        if step >= total_steps:
            break

    plt.figure(figsize=(10, 6))
    plt.plot(history['step'], history['train'], label='Train (dynamic mixtures)', marker='o', markersize=3)
    plt.plot(history['step'], history['val'], label='Val (held-out utterances)', marker='s', markersize=3)
    plt.xlabel('Step')
    plt.ylabel('Negative SI-SDR (dB)')
    plt.title('Module 1: Conv-TasNet Separation — Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/loss_curves/separation_loss.png', dpi=150)
    plt.close()
    print(f"\nBest val loss {best_val:.3f} (= {-best_val:.2f} dB SI-SDR). Training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Conv-TasNet separation model')
    parser.add_argument('--config', default='configs/separation.yaml', help='Config file path')
    parser.add_argument('--threads', type=int, default=None, help='CPU threads for torch')
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    train(args.config)
