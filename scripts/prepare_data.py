"""
Data Preparation — builds every dataset used by the project from LibriSpeech dev-clean.

NOTE: All three modules use LibriSpeech dev-clean (40 speakers, ~5.4 h, 16 kHz).
VoxCeleb (registration required) and VCTK (~10 GB) are NOT used. Report results
as "LibriSpeech dev-clean", not as VoxCeleb/VCTK.

Speaker split (shared by all modules, gender-balanced, seeded):
  - 30 "train" speakers: used for training all models
  - 10 "test" speakers: never seen in training; used for every test metric
Within each train speaker, utterances are further split 90/10 into train/val.

Produces:
  data/manifest.json                     speaker split + utterance lists
  data/mixtures/{val,test}/{mix_clean,s1,s2}/   fixed 2-speaker mixtures
      val  = train speakers' val utterances (model selection)
      test = held-out test speakers (reported SI-SDRi)
  data/diarization_test/conv_XXXX.wav + .csv    turn-taking conversations from
      held-out speakers with ground-truth (speaker, start, end) for DER
  samples/                               demo clips for the Streamlit app

Separation training uses on-the-fly mixing of train-speaker utterances, so no
training mixtures are written to disk.
"""

import json
import os
import random
import shutil
import sys

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
from training.data_utils import SAMPLE_RATE, rms_normalize  # noqa: E402

SEED = 42
NUM_TEST_SPEAKERS_PER_GENDER = 5
VAL_FRACTION = 0.1
MIN_UTT_SEC = 1.0
MIX_MAX_SEC = 4.0


def ensure_librispeech_downloaded(data_root):
    ls_dir = os.path.join(data_root, "librispeech_raw", "LibriSpeech", "dev-clean")
    if os.path.isdir(ls_dir) and os.listdir(ls_dir):
        return ls_dir
    print("Downloading LibriSpeech dev-clean via torchaudio...")
    import torchaudio
    torchaudio.datasets.LIBRISPEECH(os.path.join(data_root, "librispeech_raw"),
                                    url="dev-clean", download=True)
    return ls_dir


def read_genders(ls_root):
    genders = {}
    with open(os.path.join(ls_root, 'SPEAKERS.TXT'), 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith(';'):
                continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 3 and parts[2] == 'dev-clean':
                genders[parts[0]] = parts[1]
    return genders


def build_manifest(ls_dir, rng):
    genders = read_genders(os.path.dirname(ls_dir))
    speakers = sorted(d for d in os.listdir(ls_dir) if os.path.isdir(os.path.join(ls_dir, d)))

    # Gender-balanced held-out test speakers
    test_speakers = []
    for g in ('F', 'M'):
        pool = sorted(s for s in speakers if genders.get(s) == g)
        test_speakers += rng.sample(pool, NUM_TEST_SPEAKERS_PER_GENDER)

    manifest = {'sample_rate': SAMPLE_RATE, 'source': 'LibriSpeech dev-clean',
                'speakers': {}, 'utterances': {}}

    for spk in tqdm(speakers, desc="Indexing speakers"):
        paths = []
        for root, _, files in os.walk(os.path.join(ls_dir, spk)):
            for f in sorted(files):
                if f.endswith('.flac'):
                    full = os.path.join(root, f)
                    if sf.info(full).duration >= MIN_UTT_SEC:
                        paths.append(os.path.relpath(full, PROJECT_ROOT).replace('\\', '/'))
        paths.sort()
        rng.shuffle(paths)
        n_val = max(1, int(len(paths) * VAL_FRACTION))
        manifest['speakers'][spk] = {
            'gender': genders.get(spk, '?'),
            'split': 'test' if spk in test_speakers else 'train',
        }
        manifest['utterances'][spk] = {'train': sorted(paths[n_val:]),
                                       'val': sorted(paths[:n_val])}
    return manifest


def _load(rel):
    audio, sr = sf.read(os.path.join(PROJECT_ROOT, rel), dtype='float32')
    assert sr == SAMPLE_RATE
    return audio


def make_mixtures(manifest, speakers, subset, out_dir, num, rng):
    """Fixed 'min'-mode 2-speaker mixtures with a random relative gain in [-5, 5] dB."""
    dirs = {k: os.path.join(out_dir, k) for k in ('mix_clean', 's1', 's2')}
    for d in dirs.values():
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)

    rows = []
    for i in tqdm(range(num), desc=f"Mixtures -> {os.path.relpath(out_dir, PROJECT_ROOT)}"):
        spk1, spk2 = rng.sample(speakers, 2)
        u1 = rng.choice(manifest['utterances'][spk1][subset])
        u2 = rng.choice(manifest['utterances'][spk2][subset])
        w1, w2 = _load(u1), _load(u2)
        length = min(len(w1), len(w2), int(MIX_MAX_SEC * SAMPLE_RATE))
        w1 = rms_normalize(w1[:length])
        w2 = rms_normalize(w2[:length]) * 10 ** (rng.uniform(-5, 5) / 20)
        mix = w1 + w2
        peak = np.abs(np.stack([mix, w1, w2])).max()
        if peak > 0.95:
            scale = 0.95 / peak
            mix, w1, w2 = mix * scale, w1 * scale, w2 * scale

        name = f"mixture_{i:04d}.wav"
        sf.write(os.path.join(dirs['mix_clean'], name), mix, SAMPLE_RATE)
        sf.write(os.path.join(dirs['s1'], name), w1, SAMPLE_RATE)
        sf.write(os.path.join(dirs['s2'], name), w2, SAMPLE_RATE)
        rows.append({'file': name, 'spk1': spk1, 'spk2': spk2, 'utt1': u1, 'utt2': u2})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'metadata.csv'), index=False)


def make_conversations(manifest, speakers, out_dir, num, rng):
    """Non-overlapping 2-speaker turn-taking clips with ground-truth segments."""
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    for i in tqdm(range(num), desc="Diarization test conversations"):
        spk_a, spk_b = rng.sample(speakers, 2)
        pools = {s: list(manifest['utterances'][s]['train'] + manifest['utterances'][s]['val'])
                 for s in (spk_a, spk_b)}
        audio, segments, t = [], [], 0.0
        num_turns = rng.randint(4, 7)
        for turn in range(num_turns):
            spk = (spk_a, spk_b)[turn % 2]
            w = rms_normalize(_load(rng.choice(pools[spk])))
            w = w[:int(rng.uniform(2.0, 5.0) * SAMPLE_RATE)]
            # Trim leading/trailing silence so the reference segment is actual speech
            w, _ = librosa.effects.trim(w, top_db=35)
            gap = np.zeros(int(rng.uniform(0.2, 0.8) * SAMPLE_RATE), dtype=np.float32)
            audio += [w, gap]
            segments.append({'speaker': spk, 'start': round(t, 3),
                             'end': round(t + len(w) / SAMPLE_RATE, 3)})
            t += (len(w) + len(gap)) / SAMPLE_RATE
        audio = np.concatenate(audio)
        audio = audio / max(1.0, np.abs(audio).max() / 0.95)
        name = f"conv_{i:04d}"
        sf.write(os.path.join(out_dir, name + '.wav'), audio, SAMPLE_RATE)
        pd.DataFrame(segments).to_csv(os.path.join(out_dir, name + '.csv'), index=False)


def make_samples(data_root, samples_dir):
    os.makedirs(samples_dir, exist_ok=True)
    for f in os.listdir(samples_dir):
        if f.startswith('sample_'):
            os.remove(os.path.join(samples_dir, f))
    test_mix = os.path.join(data_root, 'mixtures', 'test', 'mix_clean')
    for f in sorted(os.listdir(test_mix))[:3]:
        shutil.copy2(os.path.join(test_mix, f), os.path.join(samples_dir, f"sample_{f}"))
    conv_dir = os.path.join(data_root, 'diarization_test')
    shutil.copy2(os.path.join(conv_dir, 'conv_0000.wav'),
                 os.path.join(samples_dir, 'sample_conversation_0000.wav'))


def main():
    rng = random.Random(SEED)
    data_root = os.path.join(PROJECT_ROOT, 'data')
    ls_dir = ensure_librispeech_downloaded(data_root)

    manifest = build_manifest(ls_dir, rng)
    with open(os.path.join(data_root, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1)

    train_spk = sorted(s for s, i in manifest['speakers'].items() if i['split'] == 'train')
    test_spk = sorted(s for s, i in manifest['speakers'].items() if i['split'] == 'test')
    print(f"Train speakers ({len(train_spk)}): {train_spk}")
    print(f"Test speakers  ({len(test_spk)}): {test_spk}")

    make_mixtures(manifest, train_spk, 'val', os.path.join(data_root, 'mixtures', 'val'), 100, rng)
    make_mixtures(manifest, test_spk, 'train', os.path.join(data_root, 'mixtures', 'test'), 200, rng)
    make_conversations(manifest, test_spk, os.path.join(data_root, 'diarization_test'), 30, rng)
    make_samples(data_root, os.path.join(PROJECT_ROOT, 'samples'))
    print("Data preparation complete.")


if __name__ == "__main__":
    main()
