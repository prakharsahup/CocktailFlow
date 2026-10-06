"""
Evaluation script for Module 2 — Speaker Embedding & Diarization

All metrics use the 10 held-out LibriSpeech dev-clean test speakers (never seen in
training):
  - Speaker verification EER: cosine scoring of same- vs different-speaker
    utterance pairs (how well embeddings generalize to new speakers)
  - Diarization Error Rate (DER) on data/diarization_test: 30 two-speaker
    turn-taking conversations with ground-truth boundaries. Frame-level (10 ms),
    no forgiveness collar, optimal speaker mapping (Hungarian), non-overlapped speech.

Outputs:
  - results/metrics/diarization_der.csv
  - results/metrics/speaker_verification.csv
"""

import os
import random
import sys

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_curve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.diarization import SAMPLE_RATE, diarize, get_embedding
from training.data_utils import load_audio, load_manifest, speakers_in, utterances

FRAME_RATE = 100  # 10 ms frames


def _to_frames(segments, num_frames):
    """List of (speaker, start, end) -> per-frame label array ('' = non-speech)."""
    frames = np.full(num_frames, '', dtype=object)
    for spk, start, end in segments:
        frames[int(round(start * FRAME_RATE)):min(int(round(end * FRAME_RATE)), num_frames)] = spk
    return frames


def compute_der(predicted, ground_truth, total_duration):
    """Diarization Error Rate with optimal one-to-one speaker mapping.

    DER = (missed speech + false alarm + speaker confusion) / total reference speech

    Args:
        predicted: list of (speaker_id, start, end) tuples (arbitrary cluster names)
        ground_truth: list of (speaker_id, start, end) tuples
        total_duration: audio duration in seconds

    Returns:
        dict with der, miss, false_alarm, confusion (fractions of reference speech)
    """
    n = int(round(total_duration * FRAME_RATE))
    ref, hyp = _to_frames(ground_truth, n), _to_frames(predicted, n)

    ref_spk = sorted(set(ref) - {''})
    hyp_spk = sorted(set(hyp) - {''})
    overlap = np.array([[np.sum((ref == r) & (hyp == h)) for h in hyp_spk] for r in ref_spk])
    mapping = {}
    if overlap.size:
        rows, cols = linear_sum_assignment(-overlap)
        mapping = {hyp_spk[c]: ref_spk[r] for r, c in zip(rows, cols)}
    hyp_mapped = np.array([mapping.get(h, '?' if h else '') for h in hyp], dtype=object)

    ref_speech = ref != ''
    hyp_speech = hyp_mapped != ''
    total = max(ref_speech.sum(), 1)
    miss = np.sum(ref_speech & ~hyp_speech) / total
    false_alarm = np.sum(~ref_speech & hyp_speech) / total
    confusion = np.sum(ref_speech & hyp_speech & (ref != hyp_mapped)) / total
    return {'der': miss + false_alarm + confusion, 'miss': miss,
            'false_alarm': false_alarm, 'confusion': confusion}


def evaluate_der(conv_dir=os.path.join('data', 'diarization_test')):
    rows = []
    files = sorted(f for f in os.listdir(conv_dir) if f.endswith('.wav'))
    for f in files:
        audio, sr = sf.read(os.path.join(conv_dir, f), dtype='float32')
        gt_df = pd.read_csv(os.path.join(conv_dir, f.replace('.wav', '.csv')), dtype={'speaker': str})
        gt = list(gt_df.itertuples(index=False, name=None))
        pred = diarize(audio, sr, num_speakers=2)
        res = compute_der(pred, gt, len(audio) / sr)
        rows.append({'file': f, 'duration_sec': round(len(audio) / sr, 2),
                     'num_pred_segments': len(pred), **{k: round(v, 4) for k, v in res.items()}})
    df = pd.DataFrame(rows)
    # Overall DER weighted by duration (standard), plus the per-file mean
    w = df['duration_sec'] / df['duration_sec'].sum()
    summary = {k: float((df[k] * w).sum()) for k in ('der', 'miss', 'false_alarm', 'confusion')}
    df.loc[len(df)] = {'file': 'OVERALL (duration-weighted)', 'duration_sec': df['duration_sec'].sum(),
                       'num_pred_segments': df['num_pred_segments'].sum(),
                       **{k: round(v, 4) for k, v in summary.items()}}
    df.to_csv('results/metrics/diarization_der.csv', index=False)
    print(f"DER on {len(files)} conversations: {100 * summary['der']:.1f}% "
          f"(miss {100 * summary['miss']:.1f}%, FA {100 * summary['false_alarm']:.1f}%, "
          f"confusion {100 * summary['confusion']:.1f}%)")
    return summary


def evaluate_eer(utts_per_speaker=20, crop_sec=3.0):
    """Speaker-verification EER on held-out speakers (all same/different pairs)."""
    manifest = load_manifest()
    rng = random.Random(0)
    embs, labels = [], []
    crop = int(crop_sec * SAMPLE_RATE)
    for spk in speakers_in(manifest, 'test'):
        for path in rng.sample(utterances(manifest, spk, 'train'), utts_per_speaker):
            audio = load_audio(path)[:crop]
            e = get_embedding(audio, SAMPLE_RATE)
            embs.append(e / np.linalg.norm(e))
            labels.append(spk)
    embs = np.stack(embs)
    labels = np.array(labels)
    scores = embs @ embs.T
    iu = np.triu_indices(len(labels), k=1)
    target = (labels[:, None] == labels[None, :])[iu]
    fpr, tpr, _ = roc_curve(target, scores[iu])
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    pd.DataFrame([{'num_speakers': len(set(labels)), 'num_trials': int(len(target)),
                   'crop_sec': crop_sec, 'eer': round(eer, 4),
                   'mean_same_speaker_cos': round(float(scores[iu][target].mean()), 4),
                   'mean_diff_speaker_cos': round(float(scores[iu][~target].mean()), 4)}]
                 ).to_csv('results/metrics/speaker_verification.csv', index=False)
    print(f"Speaker verification EER (unseen speakers, {crop_sec:.0f}s clips): {100 * eer:.1f}%")
    return eer


if __name__ == '__main__':
    os.makedirs('results/metrics', exist_ok=True)
    evaluate_eer()
    evaluate_der()
