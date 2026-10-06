# Cocktail Party Problem — Phase 1

An integrated deep learning pipeline: **speech separation → speaker diarization → voice conversion**, combined into a locally-hosted Streamlit demo. Everything trains on CPU.

## Team
- Prakhar Sahu (Separation)
- Sanmay Anand (Diarization)
- Hriday Jadhav (Voice Conversion)
- Ankush Pratham (Integration)

## Quick Start

```bash
pip install -r requirements.txt            # Python 3.10
python scripts/prepare_data.py             # downloads LibriSpeech dev-clean if missing
python scripts/train_all.py                # trains + evaluates all modules (~3 h on CPU)
streamlit run app.py
```

The 16 kHz HiFi-GAN vocoder is downloaded automatically on first use (into `checkpoints/hifigan-libritts-16kHz/`).

## Architecture

```
Mixture audio → [Module 1: Conv-TasNet Separation]
                    → stream1, stream2
                        → [Module 2: ECAPA-TDNN speaker embeddings + diarization]
                            → speaker timeline + embeddings
                                → [Module 3: AutoVC voice conversion + HiFi-GAN vocoder]
                                    → converted waveform → [Pitch Modulation DSP]
```

Module 3 uses Module 2's trained, frozen embedding network as its speaker encoder. The same network also scores speaker similarity.

## Data

All three modules use **LibriSpeech dev-clean** (40 speakers, ~5.4 h, 16 kHz). VoxCeleb and VCTK are **not** used. `scripts/prepare_data.py` creates one gender-balanced speaker split, saved in `data/manifest.json`, that every module shares:

| Split | Speakers | Used for |
|---|---|---|
| train | 30 (90% of each speaker's utterances) | training all models |
| val | same 30 (other 10% of utterances) | model selection |
| test | 10 held-out speakers | **every reported test metric** |

- **Separation** trains on 2-speaker mixtures created on the fly from the training speakers, with a random relative gain of −5 to +5 dB.
- **Test sets** are fixed:
  - `data/mixtures/test`: 200 mixtures of up to 4 s each.
  - `data/diarization_test`: 30 turn-taking conversations with ground-truth segments.

## Results (CPU training)

| Module | Model (size) | Training | Test metric (held-out speakers) |
|---|---|---|---|
| Separation | Conv-TasNet, 2.5 M params (N=256, L=32, B=128, H=256, X=8, R=3) | 4000 steps × batch 4, 81 min | **SI-SDRi 3.7 dB** mean / 3.8 dB median (200 mixtures; 82% improved) |
| Diarization | ECAPA-TDNN, 1.7 M params (C=256), AAM-Softmax | 20 epochs, ~13 min | **EER 10.9%** (speaker verification, 3 s clips); **DER 14.4%** (miss 1.5%, FA 3.8%, confusion 9.1%); closed-set val accuracy 100% |
| Voice conversion | AutoVC (dim_neck=16, freq=16) + pretrained HiFi-GAN 16 kHz | 8000 steps × batch 16, 66 min | speaker similarity converted→target **0.62** (vs. 0.12 source→target), 95% of conversions closer to target than source (seen speakers); **0.30**, 70% for unseen speakers |

Notes on reading these numbers:
- **Separation is under-trained.** Validation SI-SDR was still rising when training stopped. Published Conv-TasNet results (~15 dB) need GPU-days of training on far more data; raise `training.steps` in `configs/separation.yaml` to improve this.
- **DER** is measured frame by frame with no forgiveness collar, on non-overlapping speech, using the best match between predicted and true speaker labels.
- **Voice-conversion similarity** is scored with the same ECAPA model that conditions the converter, which flatters the result. Listen to `results/audio_examples/vc_*` and run the MOS survey. Unseen-speaker (zero-shot) conversion is weak with only 30 training speakers.

Outputs:
- `results/metrics/`: per-file CSVs and the t-SNE plot.
- `results/loss_curves/`: training curves.
- `results/audio_examples/`: example audio.

## Running the Demo

```bash
streamlit run app.py
```

- `samples/sample_mixture_*.wav`: two overlapping speakers. Run Separate, then *Analyse Speakers* (per-stream speech timeline and how similar the two streams sound), then Convert.
- `samples/sample_conversation_0000.wav`: two speakers taking turns. Use *Diarize Input*.

## Tests

```bash
python -m pytest tests -q
```

The tests that use real models are skipped until the checkpoints exist.

## Pipeline Sample Rate

All modules operate at **16 kHz**.

## References

- Luo & Mesgarani, Conv-TasNet, arXiv:1809.07454
- Desplanques et al., ECAPA-TDNN, arXiv:2005.07143
- Qian et al., AutoVC, arXiv:1905.05879
- Original build spec: `README (1).md`. It describes the planned VoxCeleb/VCTK setup; this README describes what is actually implemented.
