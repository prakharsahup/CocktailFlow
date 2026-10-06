"""
Shared data helpers for all three modules.

Every module reads the same manifest produced by scripts/prepare_data.py, so the
speaker split (train speakers vs. held-out test speakers) is identical everywhere
and no test speaker or utterance ever leaks into training.
"""

import json
import os
import random

import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MANIFEST_PATH = os.path.join(PROJECT_ROOT, 'data', 'manifest.json')
SAMPLE_RATE = 16000


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_manifest(path=MANIFEST_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/prepare_data.py` first.")
    with open(path, 'r') as f:
        return json.load(f)


def abs_path(rel_path):
    return os.path.join(PROJECT_ROOT, rel_path)


def load_audio(rel_path):
    """Load a 16 kHz mono file listed in the manifest as float32."""
    audio, sr = sf.read(abs_path(rel_path), dtype='float32')
    if audio.ndim > 1:
        audio = audio[:, 0]
    assert sr == SAMPLE_RATE, f"{rel_path}: expected {SAMPLE_RATE} Hz, got {sr}"
    return audio


def speakers_in(manifest, split):
    """Speaker ids belonging to 'train' or 'test'."""
    return sorted(s for s, info in manifest['speakers'].items() if info['split'] == split)


def utterances(manifest, speaker, subset):
    """Utterance paths for a speaker; subset is 'train' or 'val' (per-speaker split)."""
    return manifest['utterances'][speaker][subset]


def random_crop(audio, length, rng=random):
    """Random crop to `length` samples, zero-padding if shorter."""
    if len(audio) > length:
        start = rng.randint(0, len(audio) - length)
        return audio[start:start + length]
    return np.pad(audio, (0, length - len(audio)))


def rms_normalize(audio, target_db=-25.0):
    rms = np.sqrt(np.mean(audio ** 2) + 1e-12)
    return audio * (10 ** (target_db / 20) / rms)
