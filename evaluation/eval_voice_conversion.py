"""
Evaluation script for Module 3 — Voice Conversion

Speaker similarity = cosine similarity of Module 2 embeddings (not a duplicate model).
For each source utterance -> target speaker conversion we report:
  - sim(converted, target)   should be high
  - sim(converted, source)   should be low
  - sim(source, target)      baseline: how similar the two speakers already are
  - success = sim(converted, target) > sim(converted, source)

The target reference is a different utterance from the one used as target in scoring,
so the score is not just "matches the conditioning clip".

Two settings:
  - seen:   train speakers, held-out (val) utterances
  - unseen: the 10 held-out test speakers (zero-shot conversion)

Caveat: the same ECAPA model conditions the converter and scores it, which favours
the converter; listen to results/audio_examples/ too (and run the MOS survey).

Outputs:
  - results/metrics/speaker_similarity.csv
  - results/audio_examples/vc_*.wav
"""

import os
import random
import sys

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.voice_conversion import SAMPLE_RATE, convert, modulate, speaker_embedding
from training.data_utils import load_audio, load_manifest, speakers_in, utterances

MAX_SRC_SEC = 6.0


def cos(a, b):
    return float(np.dot(a, b))  # embeddings are unit-norm


def pick_pairs(manifest, setting, num_pairs, rng):
    if setting == 'seen':
        speakers, subset = speakers_in(manifest, 'train'), 'val'
    else:
        speakers, subset = speakers_in(manifest, 'test'), 'train'
    pairs = []
    for _ in range(num_pairs):
        src_spk, tgt_spk = rng.sample(speakers, 2)
        src = rng.choice(utterances(manifest, src_spk, subset))
        ref, score_ref = rng.sample(utterances(manifest, tgt_spk, subset), 2)
        pairs.append((src_spk, tgt_spk, src, ref, score_ref))
    return pairs


def evaluate(num_pairs=20, num_examples=4):
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/audio_examples', exist_ok=True)
    manifest = load_manifest()
    genders = {s: i['gender'] for s, i in manifest['speakers'].items()}
    rng = random.Random(0)

    rows = []
    for setting in ('seen', 'unseen'):
        for k, (src_spk, tgt_spk, src_p, ref_p, score_p) in enumerate(
                pick_pairs(manifest, setting, num_pairs, rng)):
            source = load_audio(src_p)[:int(MAX_SRC_SEC * SAMPLE_RATE)]
            ref, score_ref = load_audio(ref_p), load_audio(score_p)
            converted = convert(source, ref, SAMPLE_RATE)

            e_conv, e_src, e_tgt = (speaker_embedding(x) for x in (converted, source, score_ref))
            row = {'setting': setting, 'source_speaker': src_spk, 'target_speaker': tgt_spk,
                   'genders': f"{genders[src_spk]}->{genders[tgt_spk]}",
                   'sim_converted_vs_target': cos(e_conv, e_tgt),
                   'sim_converted_vs_source': cos(e_conv, e_src),
                   'sim_source_vs_target': cos(e_src, e_tgt)}
            row['success'] = row['sim_converted_vs_target'] > row['sim_converted_vs_source']
            rows.append(row)

            if k < num_examples:
                name = f"vc_{setting}_{k}_{src_spk}_to_{tgt_spk}"
                sf.write(f'results/audio_examples/{name}_source.wav', source, SAMPLE_RATE)
                sf.write(f'results/audio_examples/{name}_target_ref.wav', ref, SAMPLE_RATE)
                sf.write(f'results/audio_examples/{name}_converted.wav', converted, SAMPLE_RATE)
                sf.write(f'results/audio_examples/{name}_converted_pitch+3.wav',
                         modulate(converted, SAMPLE_RATE, 3.0), SAMPLE_RATE)

    df = pd.DataFrame(rows)
    df.to_csv('results/metrics/speaker_similarity.csv', index=False, float_format='%.4f')
    summary = df.groupby('setting')[['sim_converted_vs_target', 'sim_converted_vs_source',
                                     'sim_source_vs_target', 'success']].mean()
    print("\nMean speaker similarity (cosine, Module 2 embeddings):")
    print(summary.round(3).to_string())
    print("\nSaved results/metrics/speaker_similarity.csv and audio to results/audio_examples/")
    print("\n--- MOS Survey Template ---")
    print("Rate each *_converted.wav on naturalness (1-5): 1 = very unnatural, 5 = natural.")
    return summary


if __name__ == '__main__':
    evaluate()
