"""
Evaluation script for Module 1 — Speech Separation

Computes SI-SDRi (Scale-Invariant SDR improvement) on data/mixtures/test: 200 mixtures
of the 10 held-out LibriSpeech dev-clean speakers (never seen in training).
Saves before/after spectrograms and example audio clips.
"""

import os
import sys
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import librosa.display

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.separation import separate, si_sdr, SAMPLE_RATE


def compute_si_sdri(estimated, reference, mixture):
    """Compute SI-SDR improvement over unprocessed mixture."""
    est_tensor = torch.FloatTensor(estimated)
    ref_tensor = torch.FloatTensor(reference)
    mix_tensor = torch.FloatTensor(mixture)

    # Truncate to same length
    min_len = min(len(est_tensor), len(ref_tensor), len(mix_tensor))
    est_tensor = est_tensor[:min_len]
    ref_tensor = ref_tensor[:min_len]
    mix_tensor = mix_tensor[:min_len]

    sdr_est = si_sdr(est_tensor.unsqueeze(0), ref_tensor.unsqueeze(0)).item()
    sdr_mix = si_sdr(mix_tensor.unsqueeze(0), ref_tensor.unsqueeze(0)).item()

    return sdr_est - sdr_mix  # improvement in dB


def save_spectrogram_comparison(mixture, separated, sr, filename):
    """Save before/after spectrogram visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    for ax, audio, title in zip(axes,
                                 [mixture, separated[0], separated[1]],
                                 ['Mixture', 'Separated Stream 1', 'Separated Stream 2']):
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=sr, hop_length=512, x_axis='time',
                                 y_axis='mel', ax=ax)
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def evaluate(test_dir=None, num_examples=5):
    """Run full evaluation on test set."""

    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/audio_examples', exist_ok=True)

    if test_dir is None:
        test_dir = os.path.join('data', 'mixtures', 'test')

    mix_dir = os.path.join(test_dir, 'mix_clean')
    s1_dir = os.path.join(test_dir, 's1')
    s2_dir = os.path.join(test_dir, 's2')

    if not os.path.exists(mix_dir):
        print(f"Test directory not found: {mix_dir}")
        print("Run scripts/prepare_data.py first.")
        return

    files = sorted(os.listdir(mix_dir))
    results = []

    for i, filename in enumerate(files):
        mix_path = os.path.join(mix_dir, filename)
        stream1, stream2 = separate(mix_path)

        # Load references
        s1, _ = librosa.load(os.path.join(s1_dir, filename), sr=SAMPLE_RATE)
        s2, _ = librosa.load(os.path.join(s2_dir, filename), sr=SAMPLE_RATE)
        mix, _ = librosa.load(mix_path, sr=SAMPLE_RATE)

        # Compute SI-SDRi for best permutation
        sisdri_p1 = (compute_si_sdri(stream1, s1, mix) + compute_si_sdri(stream2, s2, mix)) / 2
        sisdri_p2 = (compute_si_sdri(stream1, s2, mix) + compute_si_sdri(stream2, s1, mix)) / 2
        sisdri = max(sisdri_p1, sisdri_p2)

        results.append({'file': filename, 'si_sdri_db': sisdri})

        # Save example audio + spectrograms
        if i < num_examples:
            stem = os.path.splitext(filename)[0]
            sf.write(f'results/audio_examples/sep_{stem}_mixture.wav', mix, SAMPLE_RATE)
            sf.write(f'results/audio_examples/sep_{stem}_stream1.wav', stream1, SAMPLE_RATE)
            sf.write(f'results/audio_examples/sep_{stem}_stream2.wav', stream2, SAMPLE_RATE)
            save_spectrogram_comparison(mix, [stream1, stream2], SAMPLE_RATE,
                                        f'results/audio_examples/sep_{stem}_spectrogram.png')

    # Save metrics
    df = pd.DataFrame(results)
    mean, median = df['si_sdri_db'].mean(), df['si_sdri_db'].median()
    df.loc[len(df)] = {'file': 'MEAN', 'si_sdri_db': mean}
    df.loc[len(df)] = {'file': 'MEDIAN', 'si_sdri_db': median}
    df.to_csv('results/metrics/separation_sisdri.csv', index=False)
    print(f"\nSI-SDRi results saved to results/metrics/separation_sisdri.csv")
    print(f"Test mixtures: {len(results)} | Mean SI-SDRi: {mean:.2f} dB | Median: {median:.2f} dB")


if __name__ == '__main__':
    evaluate()
