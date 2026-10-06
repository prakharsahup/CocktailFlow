# Cocktail Party Problem: Complete Project Guide

This guide explains the whole project: what it does, how the data flows, every model and its architecture, all the maths, how training and evaluation work, and how the demo app ties it together. It was written by reading every source file, config, log and metric in this folder.

---

## Table of contents

1. [What the project does](#1-what-the-project-does)
2. [The big picture: pipeline and folder map](#2-the-big-picture-pipeline-and-folder-map)
3. [Two backends: "pretrained" vs "custom"](#3-two-backends-pretrained-vs-custom)
4. [Audio and signal-processing foundations (the maths everything uses)](#4-audio-and-signal-processing-foundations)
5. [Data preparation](#5-data-preparation)
6. [Module 1: Speech separation (Conv-TasNet)](#6-module-1-speech-separation-conv-tasnet)
7. [Module 2: Speaker embeddings and diarization (ECAPA-TDNN)](#7-module-2-speaker-embeddings-and-diarization-ecapa-tdnn)
8. [Module 3: Voice conversion (AutoVC + HiFi-GAN, and OpenVoice v2)](#8-module-3-voice-conversion)
9. [Pitch modulation (DSP)](#9-pitch-modulation-dsp)
10. [Optimisers and learning-rate schedules](#10-optimisers-and-learning-rate-schedules)
11. [Evaluation metrics, with formulas](#11-evaluation-metrics)
12. [The Streamlit app, step by step](#12-the-streamlit-app)
13. [Results](#13-results)
14. [Tests](#14-tests)
15. [How to run everything](#15-how-to-run-everything)
16. [Known issues and limitations](#16-known-issues-and-limitations)
17. [Glossary](#17-glossary)
18. [References](#18-references)

---

## 1. What the project does

The **cocktail party problem** is the human ability to follow one voice in a noisy room full of talkers. This project rebuilds a simple version of it as a three-stage deep-learning pipeline:

| Stage | Question it answers | Model |
|---|---|---|
| **Module 1: Separation** | "Two people are talking over each other. Give me each voice on its own." | Conv-TasNet |
| **Module 2: Diarization** | "Who is speaking, and when? Are these two voices different people?" | ECAPA-TDNN speaker embeddings + clustering |
| **Module 3: Voice conversion** | "Make speaker A's words sound like speaker B's voice." Then optionally shift the pitch. | AutoVC + HiFi-GAN (custom) or OpenVoice v2 (pretrained) |

Everything runs at **16 kHz mono** and was trained on a **CPU**. All data comes from **LibriSpeech dev-clean**: 40 speakers, about 5.4 hours of read English audiobooks.

**Team** (from `README.md`):
- Prakhar Sahu: Separation
- Sanmay Anand: Diarization
- Hriday Jadhav: Voice Conversion
- Ankush Pratham: Integration

---

## 2. The big picture: pipeline and folder map

### 2.1 Data flow

```
             ┌─────────────────────────────┐
mixture.wav ─▶  Module 1: Conv-TasNet       │──▶ stream 1 (speaker A)
 (A + B)     │  separate_waveform()        │──▶ stream 2 (speaker B)
             └─────────────────────────────┘
                              │
                              ▼
             ┌─────────────────────────────┐
             │  Module 2: ECAPA-TDNN        │──▶ 192-dim speaker embedding per stream
             │  get_embedding(), diarize()  │──▶ "who spoke when" timeline
             └─────────────────────────────┘
                              │  (embeddings are the speaker identity)
                              ▼
             ┌─────────────────────────────┐
             │  Module 3: voice conversion  │──▶ A's words in B's voice
             │  convert()                   │
             └─────────────────────────────┘
                              │
                              ▼
             ┌─────────────────────────────┐
             │  Pitch shift (DSP)           │──▶ final audio, ±12 semitones
             │  modulate()                  │
             └─────────────────────────────┘
```

The modules connect through a small **interface contract**: each `models/*.py` file exposes a few public functions, and the app and evaluation scripts only call those.

| Function | File | Signature → output |
|---|---|---|
| `separate(path)` | `models/separation.py` | file path → `(stream1, stream2)` numpy arrays at 16 kHz |
| `separate_waveform(audio)` | `models/separation.py` | mono 16 kHz array → `(stream1, stream2)` |
| `get_embedding(audio, sr)` | `models/diarization.py` | audio → 192-dim vector |
| `diarize(audio, sr, num_speakers)` | `models/diarization.py` | audio → list of `(speaker_id, start_sec, end_sec)` |
| `speech_activity(audio, sr)` | `models/diarization.py` | audio → boolean array, one entry per 10 ms frame |
| `convert(source, target_ref, sr)` | `models/voice_conversion.py` | source audio + target reference → converted audio, same length as the source |
| `modulate(audio, sr, semitones)` | `models/voice_conversion.py` | audio → pitch-shifted audio |

**The cross-module link:** in custom mode, Module 3's speaker encoder *is* Module 2. `models/voice_conversion.py` imports `get_embedding` from `models/diarization.py` and uses it as the speaker identity that conditions AutoVC. The same embedding also scores how well a conversion worked.

### 2.2 Folder map

```
P:\ML Project
├── app.py                    Streamlit demo (the user-facing app)
├── README.md                 Short project README
├── README (1).md             Original build specification (the plan, not what was built)
├── PROJECT_GUIDE.md          This file
├── requirements.txt          Pinned Python dependencies
├── configs/
│   ├── separation.yaml       Module 1 settings (backend, architecture, training)
│   ├── diarization.yaml      Module 2 settings (+ diarization windowing)
│   └── voice_conversion.yaml Module 3 settings
├── models/                   Model definitions + public API (the "interface contract")
│   ├── separation.py         Conv-TasNet, SI-SDR, PIT loss, separate()
│   ├── diarization.py        ECAPA-TDNN, AAM-Softmax, VAD, diarize()
│   └── voice_conversion.py   AutoVC, losses, HiFi-GAN wrapper, OpenVoice wrapper, convert(), modulate()
├── training/
│   ├── data_utils.py         Shared helpers: manifest, audio loading, RMS normalisation, seeding
│   ├── train_separation.py   Trains Conv-TasNet
│   ├── train_diarization.py  Trains ECAPA-TDNN (+ --tsne plot)
│   └── train_voice_conversion.py  Trains AutoVC
├── evaluation/
│   ├── eval_separation.py    SI-SDRi on 200 test mixtures
│   ├── eval_diarization.py   EER + DER
│   └── eval_voice_conversion.py  Speaker-similarity scores + example audio
├── scripts/
│   ├── prepare_data.py       Downloads LibriSpeech, builds the speaker split and test sets
│   └── train_all.py          Runs the whole train + evaluate pipeline
├── tests/                    pytest unit tests (39 tests)
├── third_party/openvoice/    Vendored OpenVoice v2 converter network (MIT licence)
├── checkpoints/              Trained (.pt) and downloaded pretrained weights
├── data/                     LibriSpeech, manifest, test mixtures, diarization conversations
├── results/                  Metrics CSVs, loss curves, t-SNE plot, audio examples
├── samples/                  Demo clips for the app
└── logs/                     Logs from the training/eval run
```

---

## 3. Two backends: "pretrained" vs "custom"

Each `configs/*.yaml` file has a `backend:` key. **All three are currently set to `pretrained`.**

| Module | `backend: pretrained` (current default) | `backend: custom` (trained in this project) |
|---|---|---|
| Separation | **Asteroid Conv-TasNet** `JorisCos/ConvTasNet_Libri2Mix_sepclean_16k`, trained on about 212 h of Libri2Mix. 5,066,929 params. | **Our Conv-TasNet**, 2,536,625 params, trained for 81 CPU-minutes → `checkpoints/separation.pt` |
| Diarization / embeddings | **SpeechBrain ECAPA-TDNN** `speechbrain/spkrec-ecapa-voxceleb`, trained on VoxCeleb 1+2 (7,205 speakers). Channels 1024, 20,767,552 params. | **Our ECAPA-TDNN**, channels 256, 1,701,792 params, 30 training speakers → `checkpoints/diarization.pt` |
| Voice conversion | **OpenVoice v2** tone-colour converter (VITS-style flow model), 32,792,226 params, vendored in `third_party/openvoice` | **Our AutoVC**, 7,436,608 params, plus the pretrained SpeechBrain **HiFi-GAN** 16 kHz vocoder → `checkpoints/voice_conversion.pt` |

**Why both exist:** the custom models are the from-scratch implementations of the papers (the learning part of the project). On only 30 speakers and CPU time they work, but they are far weaker than the same architectures trained on hundreds of hours. The pretrained backends give the demo app good quality. For separation and diarization they use the **same architecture family** as the custom code (Conv-TasNet and ECAPA-TDNN), and both diarization variants output **192-dim** embeddings, so Module 3 works with either.

Switching is one line in a config, e.g. `backend: custom` in `configs/separation.yaml`.

---

## 4. Audio and signal-processing foundations

These ideas come up in every module.

### 4.1 Digital audio

A sound is stored as a sequence of samples $x[n]$, $n = 0, 1, \dots$, taken $f_s = 16{,}000$ times per second. One second is 16,000 numbers in roughly $[-1, 1]$. The highest frequency a 16 kHz signal can represent (the Nyquist frequency) is $f_s/2 = 8$ kHz, which is why the mel filters stop at `fmax=8000`.

### 4.2 Decibels and loudness

Ratios of energy are measured in decibels:

$$
\text{dB} = 10\log_{10}\frac{P_1}{P_2} = 20\log_{10}\frac{A_1}{A_2}
$$

where $P$ is power and $A$ is amplitude. So multiplying an amplitude by $10^{g/20}$ changes its level by $g$ dB. The code uses this in many places, e.g. `w2 * 10 ** (rng.uniform(-5, 5) / 20)` scales a voice by a random −5 to +5 dB.

**RMS normalisation** (`training/data_utils.py: rms_normalize`) sets every clip to the same loudness, −25 dBFS:

$$
\text{RMS}(x) = \sqrt{\frac{1}{N}\sum_{n=1}^{N} x[n]^2 + 10^{-12}},
\qquad
x_{\text{norm}} = x \cdot \frac{10^{-25/20}}{\text{RMS}(x)}
$$

### 4.3 Short-Time Fourier Transform (STFT)

Speech changes over time, so we analyse it in short overlapping windows. With window $w$ (Hann), window length $W$, hop $H$ and FFT size $K$:

$$
X[t, k] = \sum_{n=0}^{W-1} x[tH + n]\, w[n]\, e^{-j 2\pi k n / K},
\qquad k = 0,\dots,K/2
$$

$|X[t,k]|$ is the **magnitude spectrogram**: how much energy there is at frequency bin $k$ in frame $t$. $|X|^2$ is the **power spectrogram**. Frequency bin $k$ corresponds to $k \cdot f_s / K$ Hz.

The project uses two STFT settings:

| Used by | FFT size $K$ | Window $W$ | Hop $H$ | Frame rate |
|---|---|---|---|---|
| ECAPA features (Module 2) | 512 | 400 (25 ms) | 160 (10 ms) | 100 frames/s |
| AutoVC / HiFi-GAN mels (Module 3) | 1024 | 1024 (64 ms) | 256 (16 ms) | 62.5 frames/s |
| OpenVoice (22.05 kHz) | 1024 | 1024 | 256 (11.6 ms) | ≈86 frames/s |

### 4.4 The mel scale and mel spectrograms

Humans hear pitch roughly linearly below about 1 kHz and logarithmically above it. A **mel filterbank** is a set of $M=80$ triangular filters spaced evenly on the mel scale. librosa's default is the **Slaney** mel scale:

$$
\text{mel}(f) =
\begin{cases}
\dfrac{3f}{200}, & f < 1000\ \text{Hz} \\[2mm]
15 + \dfrac{27\,\ln(f/1000)}{\ln 6.4}, & f \ge 1000\ \text{Hz}
\end{cases}
$$

The mel spectrogram multiplies the spectrogram by the filterbank matrix $\mathbf{F}\in\mathbb{R}^{80\times(K/2+1)}$:

$$
S_{\text{mel}}[m, t] = \sum_k F[m,k]\,|X[t,k]|^p
$$

with $p=2$ (power) for ECAPA features and $p=1$ (magnitude) for AutoVC/HiFi-GAN.

**Log compression.** Loudness perception is roughly logarithmic, and logs tame the huge dynamic range.

- **Module 2** (`extract_log_mel`):
  1. Convert to decibels relative to the loudest bin: $10\log_{10}(S/\max S)$, clipped to an 80 dB range (librosa's default `top_db=80`).
  2. Apply cepstral mean normalisation (CMN): subtract each mel bin's mean over time,
     $$\tilde S[m,t] = S_{\text{dB}}[m,t] - \frac{1}{T}\sum_{t'} S_{\text{dB}}[m,t'].$$
     This removes constant microphone/channel colouration and overall loudness.
- **Module 3** (`audio_to_mel`): natural log with a floor, $\ln(\max(S, 10^{-5}))$. Silence becomes $\ln 10^{-5} \approx -11.51$ (`LOG_MEL_FLOOR`). These exact settings must match what the pretrained HiFi-GAN was trained on, otherwise it outputs noise.

### 4.5 Convolutions used everywhere

A **1-D convolution** with kernel size $k$, dilation $d$, $C_{\text{in}}$ input channels and $C_{\text{out}}$ output channels:

$$
y_c[t] = b_c + \sum_{c'=1}^{C_{\text{in}}}\sum_{i=0}^{k-1} W_{c,c',i}\; x_{c'}[t + d\,(i - \lfloor k/2 \rfloor)]
$$

- **Dilation** $d$ spaces the taps apart, so the layer "sees" $(k-1)d+1$ frames with only $k$ weights.
- **Stride** $s$ keeps every $s$-th output (downsampling).
- A **transposed convolution** does the reverse: it upsamples by inserting outputs $s$ apart and overlap-adding.
- A **1×1 convolution** ($k=1$) mixes channels at each time step. It is a per-frame linear layer.
- A **depthwise convolution** (`groups=C`) filters each channel separately with its own kernel.

### 4.6 Activations and normalisation

| Name | Formula |
|---|---|
| ReLU | $\max(0, x)$ |
| PReLU | $\max(0,x) + a\min(0,x)$, with $a$ learned |
| Leaky ReLU (HiFi-GAN, slope 0.1) | $\max(0,x) + 0.1\min(0,x)$ |
| Sigmoid | $\sigma(x) = 1/(1+e^{-x})$ |
| Tanh | $\tanh(x) = (e^{x}-e^{-x})/(e^{x}+e^{-x})$ |
| Softmax | $\text{softmax}(z)_i = e^{z_i}/\sum_j e^{z_j}$ |

**Batch normalisation** (`BatchNorm1d`) normalises each channel using statistics over the batch and time:

$$
\hat x = \frac{x - \mu_{\mathcal B}}{\sqrt{\sigma^2_{\mathcal B} + \epsilon}}\,\gamma + \beta
$$

At inference it uses running averages instead of batch statistics.

**Global layer norm (gLN)**, used in Conv-TasNet and implemented as `GroupNorm(1, C)`, normalises each example over *all* channels and time:

$$
\text{gLN}(\mathbf F) = \frac{\mathbf F - \mathbb E[\mathbf F]}{\sqrt{\text{Var}[\mathbf F]+\epsilon}}\odot\gamma + \beta,
\qquad \mathbb E,\ \text{Var over } (C, T)
$$

### 4.7 Cosine similarity

Speaker embeddings are compared by the angle between them:

$$
\cos(\mathbf a, \mathbf b) = \frac{\mathbf a\cdot\mathbf b}{\lVert\mathbf a\rVert\,\lVert\mathbf b\rVert}\in[-1,1]
$$

The value is 1 for the same direction (same voice) and about 0 for unrelated voices. If the vectors are first L2-normalised ($\mathbf a/\lVert\mathbf a\rVert$), cosine similarity is just a dot product.

---

## 5. Data preparation

**File:** `scripts/prepare_data.py`. It is seeded (`SEED = 42`), so the results are reproducible.

### 5.1 Download

If `data/librispeech_raw/LibriSpeech/dev-clean` is missing, the script downloads LibriSpeech dev-clean with `torchaudio.datasets.LIBRISPEECH`. Files are 16 kHz FLAC.

### 5.2 One shared speaker split → `data/manifest.json`

1. Read each speaker's gender from `SPEAKERS.TXT`.
2. Pick **5 female + 5 male = 10 test speakers** at random. These are never used in training.
3. The other **30 speakers are "train" speakers**.
4. For every speaker, keep utterances of at least 1 s, shuffle them, and split them **90% train / 10% val**.

| Split | Speakers | Used for |
|---|---|---|
| train | 30 (90% of their utterances) | training all models |
| val | the same 30 (the other 10%) | choosing the best checkpoint |
| test | 10 unseen speakers | every reported test metric |

The test speakers are 1462, 1988, 2078, 2086, 2412, 2428, 3081, 5694, 6345 and 777 (from `logs/prepare_data.log`). Every module reads the same manifest through `training/data_utils.py`, so **no test speaker ever leaks into training**.

### 5.3 Fixed 2-speaker mixtures → `data/mixtures/{val,test}/`

Each mixture is built from two different speakers:

$$
L = \min(\text{len}(w_1), \text{len}(w_2), 4\,\text{s}),\quad
w_1 \leftarrow \text{RMSnorm}(w_1[:L]),\quad
w_2 \leftarrow \text{RMSnorm}(w_2[:L])\cdot 10^{g/20},\ g\sim\mathcal U(-5,5)\ \text{dB}
$$

$$
\text{mix} = w_1 + w_2
$$

If any of mix, $w_1$, $w_2$ peaks above 0.95, all three are scaled by the same factor, so the relationship $\text{mix}=w_1+w_2$ is kept exactly. This is "min" mode: the mixture is as long as the shorter clip.

- `val`: 100 mixtures from the train speakers' val utterances.
- `test`: 200 mixtures from the 10 test speakers.

Each split folder contains `mix_clean/`, `s1/` and `s2/` (the ground-truth sources), plus `metadata.csv`.

### 5.4 Diarization conversations → `data/diarization_test/`

There are 30 turn-taking clips. Each clip has two test speakers alternating for 4–7 turns:
- Each turn is 2–5 s of one utterance, with leading and trailing silence trimmed (`librosa.effects.trim`, 35 dB).
- Turns are separated by a 0.2–0.8 s silent gap.
- The exact `(speaker, start, end)` of every turn is written to `conv_XXXX.csv`. These are the ground-truth labels.

There is **no overlapping speech** here, by design.

### 5.5 Demo samples → `samples/`

The first 3 test mixtures are copied as `sample_mixture_000X.wav`, and `conv_0000.wav` is copied as `sample_conversation_0000.wav`.

---

## 6. Module 1: Speech separation (Conv-TasNet)

**Files:**
- `models/separation.py`
- `training/train_separation.py`
- `evaluation/eval_separation.py`
- `configs/separation.yaml`

**Paper:** Luo & Mesgarani, *Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation*, 2019.

### 6.1 The problem

We observe one microphone signal that is the sum of two voices:

$$
x[n] = s_1[n] + s_2[n]
$$

and want estimates $\hat s_1, \hat s_2$. With a single channel there is no spatial information, so the network must learn what a voice sounds like.

### 6.2 Key idea: a learned encoder instead of an STFT, plus masks

Older systems masked an STFT. Conv-TasNet replaces the STFT with a **learned** filterbank and works entirely in the time domain:

```
x (1 × T samples)
  │  Encoder: Conv1d(1→N, kernel L, stride L/2) + ReLU
  ▼
w (N × F frames)                        ← "learned spectrogram"
  │  Separator (TCN) → masks m₁, m₂ ∈ [0,∞)^{N×F}
  ▼
d_i = w ⊙ m_i                           ← keep only speaker i's parts
  │  Decoder: ConvTranspose1d(N→1, kernel L, stride L/2)
  ▼
ŝ_i (1 × T samples)
```

### 6.3 Encoder

The encoder takes overlapping segments of $L$ samples with hop $L/2$ and projects each onto $N$ learned basis functions:

$$
\mathbf w_f = \text{ReLU}(\mathbf U\,\mathbf x_f),\qquad \mathbf U\in\mathbb R^{N\times L}
$$

In the custom config, $L=32$ samples (2 ms), so the stride is 16 samples (1 ms) and the frame rate is **1000 frames/s**, with $N=256$. The number of frames for $T$ samples is $F = (T - L)/(L/2) + 1$. `ConvTasNet.forward` right-pads the input so this divides exactly, and crops the output back to $T$.

### 6.4 Separator: Temporal Convolutional Network (TCN)

1. **Input:** gLN, then a 1×1 conv from $N \to B$ (the bottleneck, $B=128$).
2. **$R \times X$ = 3 × 8 = 24 conv blocks** (`DepthwiseSeparableConv`). Block $l$ in each stack has dilation $2^{l}$, $l=0..7$, i.e. 1, 2, 4, …, 128. Each block computes:
   $$
   \begin{aligned}
   \mathbf h &= \text{gLN}(\text{PReLU}(\text{Conv}_{1\times1}^{B\to H}(\mathbf y))) \\
   \mathbf h &= \text{gLN}(\text{PReLU}(\text{DepthwiseConv}_{P=3,\ d=2^l}(\mathbf h))) \\
   \mathbf y_{\text{next}} &= \mathbf y + \text{Conv}_{1\times1}^{H\to B}(\mathbf h) \quad\text{(residual path)}\\
   \mathbf s_{\text{block}} &= \text{Conv}_{1\times1}^{H\to B}(\mathbf h) \quad\text{(skip path)}
   \end{aligned}
   $$
   with $H=256$ hidden channels.
3. **Output:** all skip outputs are summed, then PReLU, a 1×1 conv $B \to N\cdot C$ (with $C=2$ sources) and a ReLU give the masks $\mathbf m_1, \mathbf m_2$.

**Why depthwise-separable convolution?** A normal conv with $H$ channels and kernel $P$ needs $H^2P$ weights. A depthwise conv plus 1×1 convs needs about $HP + H^2$, which is much cheaper for the same receptive field.

**Receptive field.** Each block with kernel $P=3$ and dilation $d$ widens the view by $(P-1)d = 2d$ frames. One stack adds $2(1+2+\dots+128) = 2\cdot255 = 510$ frames, and three stacks add $1530$. The separator therefore sees

$$
1 + 3\cdot 2\cdot(2^8-1) = 1531\ \text{frames} \approx 1.53\ \text{s}
$$

of context at 1 ms per frame. That is long enough to track a voice across syllables.

### 6.5 Mask, decode

$$
\mathbf d_i = \mathbf w \odot \mathbf m_i,\qquad
\hat{\mathbf s}_i = \text{ConvTranspose1d}(\mathbf d_i) \quad(\text{overlap-add of } \mathbf V^\top \mathbf d_{i,f})
$$

The decoder basis $\mathbf V\in\mathbb R^{N\times L}$ is learned separately from the encoder basis $\mathbf U$.

**Custom model hyperparameters** (paper notation, from `configs/separation.yaml`):

| Symbol | Meaning | Ours | Paper |
|---|---|---|---|
| N | encoder filters | 256 | 512 |
| L | encoder kernel (samples) | 32 (= 2 ms at 16 kHz) | 16 at 8 kHz (= 2 ms) |
| B | bottleneck channels | 128 | 128 |
| H | hidden channels in conv blocks | 256 | 512 |
| P | depthwise kernel | 3 | 3 |
| X | blocks per stack | 8 | 8 |
| R | stacks | 3 | 3 |
| mask activation | | ReLU | ReLU |
| **Params** | | **2,536,625** | ≈5.1 M |

The pretrained Asteroid model uses N=512, L=32, stride 16, B=128, H=512, X=8, R=3, gLN and a ReLU mask, for 5,066,929 params. That is essentially the paper's configuration at 16 kHz.

### 6.6 Loss: SI-SDR with permutation-invariant training

**SI-SDR (scale-invariant signal-to-distortion ratio).** It measures how much of the estimate is the true signal versus error, ignoring overall volume. First make both signals zero-mean, then project the estimate onto the reference:

$$
\mathbf s_{\text{target}} = \frac{\langle \hat{\mathbf s}, \mathbf s\rangle}{\lVert\mathbf s\rVert^2}\,\mathbf s,
\qquad
\mathbf e_{\text{noise}} = \hat{\mathbf s} - \mathbf s_{\text{target}}
$$

$$
\text{SI-SDR}(\hat{\mathbf s}, \mathbf s) = 10\log_{10}\frac{\lVert\mathbf s_{\text{target}}\rVert^2}{\lVert\mathbf e_{\text{noise}}\rVert^2}\ \ \text{dB}
$$

Scaling $\hat{\mathbf s}$ by any constant $\alpha\neq0$ scales both $\mathbf s_{\text{target}}$ and $\mathbf e_{\text{noise}}$ by $\alpha$, so the ratio is unchanged. This is why the network's output volume is arbitrary, and why `separate_waveform` rescales each stream to the mixture's peak level afterwards.

Reference points: a perfect estimate gives $+\infty$ (capped by the $10^{-8}$ epsilon), and an orthogonal (unrelated) signal gives a very negative value. The tests check both.

**Permutation problem.** The network has two outputs, but nothing says output 1 must be "speaker A". If the loss assumed a fixed order, the network would be punished for a correct separation with the labels swapped. **Utterance-level PIT** tries both assignments and keeps the better one:

$$
\mathcal L_{\text{PIT}} = -\frac{1}{2}\max\Big(
\underbrace{\text{SI-SDR}(\hat s_1,s_1)+\text{SI-SDR}(\hat s_2,s_2)}_{\text{permutation 1}},\ 
\underbrace{\text{SI-SDR}(\hat s_1,s_2)+\text{SI-SDR}(\hat s_2,s_1)}_{\text{permutation 2}}
\Big)
$$

The loss is averaged over the batch. Minimising $-\text{SI-SDR}$ maximises SI-SDR, so a training loss of −5.68 means an average SI-SDR of 5.68 dB.

### 6.7 Training (`training/train_separation.py`)

- **Dynamic mixing:** there is no fixed training set. Every step draws 2 random train speakers and a random utterance from each, then:
  1. Takes a random 2 s crop of each. Crops that land in silence (peak below $10^{-4}$) are skipped.
  2. RMS-normalises both to −25 dBFS and scales the second by $\mathcal U(-5,5)$ dB.
  3. Multiplies both by a random overall gain of $\mathcal U(-6,6)$ dB.
  4. Mixes them: mix = sum.

  With only about 4 h of audio, this gives effectively unlimited variety.
- **Batch:** 4 mixtures × 2 s.
- **Optimiser:** Adam with lr = $10^{-3}$, weight decay 0, gradient-norm clipping at 5.0.
- **Schedule:** `ReduceLROnPlateau` (factor 0.5, patience 2) on the validation loss.
- **Steps:** 4000, with validation every 500 steps on the 100 fixed val mixtures. The checkpoint with the best validation loss is saved to `checkpoints/separation.pt`.
- **Logged run:** the validation loss went from −2.21 at step 500 (2.2 dB SI-SDR) to **−5.68** at step 4000 (5.68 dB SI-SDR, 81 min on CPU). It was still improving when training stopped, so the model is under-trained.

### 6.8 Inference (`separate_waveform`)

1. The waveform has shape (1, 1, T); run the model to get (1, 2, T).
2. For each stream $s$: subtract its mean, then scale it to the mixture's peak:
   $$s \leftarrow s\cdot\frac{\max|x|}{\max|s|+10^{-8}}$$
   This makes it audible and prevents clipping.

### 6.9 Evaluation: SI-SDR improvement (`evaluation/eval_separation.py`)

$$
\text{SI-SDRi} = \text{SI-SDR}(\hat s, s) - \text{SI-SDR}(x, s)
$$

This is how much better the separated stream is than doing nothing, i.e. using the mixture itself as the estimate. It is computed for both permutations, averaged over the two sources, and the better permutation is kept. The results are written to `results/metrics/separation_sisdri.csv`, and the first 5 mixtures get example WAVs and spectrogram plots.

---

## 7. Module 2: Speaker embeddings and diarization (ECAPA-TDNN)

**Files:**
- `models/diarization.py`
- `training/train_diarization.py`
- `evaluation/eval_diarization.py`
- `configs/diarization.yaml`

**Paper:** Desplanques et al., *ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification*, 2020.

### 7.1 Goal: a "voiceprint"

We want a function $f:\text{audio}\to\mathbb R^{192}$ such that clips from the same person point in nearly the same direction, and clips from different people point in different directions, **even for people never seen in training**. Diarization, speaker verification and the voice-conversion speaker identity all build on this one embedding.

### 7.2 Input features

These are the log-mel features from §4.4: 80 mels, 25 ms window, 10 ms hop, dB scale, with per-utterance mean normalisation. A 2 s crop gives a tensor of shape (80, 201).

The pretrained SpeechBrain model computes its own 80-dim filterbank features internally with sentence-level mean normalisation, so it is given raw waveforms instead.

### 7.3 Architecture

```
log-mel (80 × T)
  │ Conv1d(80→C, k=5) + ReLU + BN                   ← initial TDNN layer
  ▼
  │ SE-Res2Net block 1 (k=3, dilation 2)  ──┐
  │ SE-Res2Net block 2 (k=3, dilation 3)  ──┤        three blocks, increasing dilation
  │ SE-Res2Net block 3 (k=3, dilation 4)  ──┤
  ▼                                          │
  Concatenate outputs of blocks 1,2,3  ◀─────┘        ← Multi-layer Feature Aggregation (MFA)
  │ Conv1d(3C→3C, k=1) + BN + ReLU
  ▼
  Attentive Statistics Pooling  (3C × T → 6C)          ← collapses time
  │ Linear(6C → 192) + BN
  ▼
192-dim speaker embedding
```

Here C = 256 for the custom model (the paper uses 512–1024; the pretrained model uses 1024). The receptive field at frame level grows through the dilated layers: the first conv sees ±2 frames, and the three blocks add $2\cdot(2+3+4)=18$ more frames, so each output frame summarises about 0.23 s before pooling.

#### (a) Res2Net block: multi-scale features

The input channels are split into $s=8$ groups of width $C/8$: $\mathbf x_1,\dots,\mathbf x_8$. Each group is convolved after adding the previous group's output:

$$
\mathbf y_1 = \mathbf x_1,\qquad
\mathbf y_2 = K_2(\mathbf x_2),\qquad
\mathbf y_i = K_i(\mathbf x_i + \mathbf y_{i-1})\quad (i\ge3)
$$

where $K_i$ = Conv(k=3, dilation $d$) → ReLU → BN. The outputs are concatenated. Later groups pass through more convolutions, so a single block mixes several receptive-field sizes cheaply.

#### (b) Squeeze-and-Excitation (SE): channel attention

The block learns which channels matter for *this* utterance:

$$
\mathbf z = \frac{1}{T}\sum_{t}\mathbf h_t \quad(\text{squeeze: average over time}),
\qquad
\mathbf a = \sigma(\mathbf W_2\,\text{ReLU}(\mathbf W_1\mathbf z)) \quad(\text{excitation})
$$

$$
\tilde{\mathbf h}_t = \mathbf a\odot\mathbf h_t
$$

$\mathbf W_1$ reduces the channels by a factor of 8 (256 → 32) and $\mathbf W_2$ expands them back.

**One full SE-Res2Net block:** 1×1 conv → BN → ReLU → Res2Net → 1×1 conv → BN → ReLU → SE, plus a residual connection from the block input.

#### (c) Multi-layer feature aggregation (MFA)

Shallow layers carry fine detail and deep layers carry abstract speaker cues. Concatenating all three blocks' outputs (3C = 768 channels) and mixing them with a 1×1 conv lets the model use both.

#### (d) Attentive statistics pooling (ASP)

A variable-length sequence $\mathbf h_1..\mathbf h_T$ has to become one fixed vector. A small attention network scores every frame **per channel**, and a softmax over time turns the scores into weights:

$$
e_{t,c} = \big[\mathbf W_b\,\text{ReLU}(\mathbf W_a\mathbf h_t + \mathbf b_a) + \mathbf b_b\big]_c,
\qquad
\alpha_{t,c} = \frac{\exp(e_{t,c})}{\sum_{\tau}\exp(e_{\tau,c})}
$$

The weighted mean and standard deviation are then:

$$
\boldsymbol\mu_c = \sum_t \alpha_{t,c}\,h_{t,c},
\qquad
\boldsymbol\sigma_c = \sqrt{\sum_t \alpha_{t,c}\,(h_{t,c}-\mu_c)^2}
$$

The output is $[\boldsymbol\mu;\boldsymbol\sigma]\in\mathbb R^{2\cdot768=1536}$. Silent or noisy frames get low weight. The std captures how the voice varies, which is also speaker-specific.

This implementation is slightly simpler than the paper's: the paper also feeds the global mean and std of $\mathbf h$ into the attention network, while this code doesn't.

#### (e) Embedding layer

$\text{Linear}(1536\to192)$ followed by BatchNorm gives the **192-dim embedding**. `EMBEDDING_DIM = 192` is a cross-module contract: AutoVC's `dim_emb` must equal it, and `AutoVC.__init__` asserts this.

**Custom model params:** 1,701,792. Pretrained SpeechBrain model: 20,767,552.

### 7.4 Loss: AAM-Softmax (Additive Angular Margin, "ArcFace")

The model is trained as a 30-class speaker classifier, but in a way that shapes the **angles** between embeddings. Let $\mathbf e$ be the embedding and $\mathbf W_j$ the learnable "centre" of class $j$. Both are L2-normalised, so

$$
\cos\theta_j = \frac{\mathbf W_j\cdot\mathbf e}{\lVert\mathbf W_j\rVert\lVert\mathbf e\rVert}.
$$

For the true class $y$, a margin $m$ is **added to the angle** before computing the logits:

$$
\text{logit}_j = s\cdot
\begin{cases}
\cos(\theta_y + m), & j = y\\
\cos\theta_j, & j\neq y
\end{cases}
\qquad
\mathcal L = -\log\frac{e^{\text{logit}_y}}{\sum_j e^{\text{logit}_j}}
$$

with $m = 0.2$ rad (about 11.5°) and $s = 30$.

**Why it works:** to classify correctly, the embedding must be closer to its own centre than to every other centre by at least the margin $m$. This forces tight clusters per speaker with clear gaps between them. Only angles matter, so the embeddings generalise to *new* speakers, whose voices land in their own regions of the sphere.

The scale $s$ sharpens the softmax, since cosines only range over [−1, 1]. Training accuracy is measured as $\arg\max_j\cos\theta_j$ (`AAMSoftmax.cosine`). After training, the class-centre matrix $\mathbf W$ is **discarded** and only the embedding network is kept. A test checks that the network has no leftover `classifier` head.

### 7.5 Training (`training/train_diarization.py`)

- **Data:** utterances from the 30 train speakers; each training sample is a random 2 s crop.
- **Augmentation:** with 50% probability, a *different* train speaker is mixed in at 10–20 dB below the main voice:
  $$\text{audio} = \text{RMSnorm}(a) + \text{RMSnorm}(b)\cdot10^{-\text{SNR}/20},\quad \text{SNR}\sim\mathcal U(10,20)$$
  This teaches the embedding to ignore the faint cross-talk that separation leaves behind.
- **Validation:** a deterministic centre 2 s crop of the val utterances.
- **Optimiser:** Adam (weight decay $2\times10^{-5}$) over the model plus the AAM centres.
- **Schedule:** **OneCycleLR** with peak lr $2\times10^{-3}$ and 15% warm-up; the formula is in §10.
- **Run length:** batch 32, configured for 30 epochs. The best checkpoint is chosen by validation accuracy and saved to `checkpoints/diarization.pt`. The logged run reached 100% closed-set validation accuracy at epoch 20 (~12.6 min).
- **`--tsne`:** draws a 2-D t-SNE plot of embeddings for the **10 unseen test speakers** (25 clips each) → `results/metrics/tsne_embeddings.png`. Well-separated colour clusters mean the embedding generalises.

**t-SNE in one line:** it places points in 2-D so that the neighbour probabilities $p_{ij}$ (a Gaussian over the cosine distances in 192-D) match the neighbour probabilities $q_{ij}$ (a Student-t in 2-D), by minimising $\text{KL}(P\Vert Q)=\sum p_{ij}\log(p_{ij}/q_{ij})$. Perplexity is 20.

### 7.6 Voice activity detection (`speech_activity`)

This is a simple energy detector at 10 ms resolution:
1. Compute the frame RMS energy (40 ms frames, 10 ms hop) and convert it to dB: $E_t = 20\log_{10}(\text{RMS}_t + 10^{-8})$.
2. Take the "loud speech" level as the 95th percentile, $P_{95}(E)$.
3. Mark a frame as speech if $E_t > P_{95}(E) - 35$ dB.

Using a percentile makes the threshold adapt to the clip's overall loudness.

### 7.7 Diarization algorithm (`diarize`)

The goal is "who spoke when" for turn-taking audio (no overlap):

1. **VAD:** speech/non-speech labels for every 10 ms frame.
2. **Sliding windows:** 1.5 s windows every 0.25 s, with the last one aligned to the end. Only windows that are **at least 50% speech** are kept.
3. **Embed** each window (in batches of 64) and **L2-normalise** the embeddings.
4. **Cluster** them with **agglomerative clustering** (cosine distance, average linkage) into `num_speakers` groups (default 2):
   1. Start with every window as its own cluster.
   2. Repeatedly merge the two clusters with the smallest average pairwise distance
      $$D(A,B) = \frac{1}{|A||B|}\sum_{a\in A}\sum_{b\in B}\big(1-\cos(\mathbf e_a,\mathbf e_b)\big)$$
   3. Stop when `num_speakers` clusters remain.
5. **Frame labelling:** each speech frame takes the cluster label of the **nearest window centre**. Non-speech frames get the label −1.
6. **Gap bridging:** a non-speech gap of at most 0.5 s between two frames with the *same* label is filled in, so one speaker's pause isn't split into two turns.
7. **Segments:** consecutive identical labels become a `(Speaker_k, start, end)` segment. Segments shorter than 0.1 s are dropped.

### 7.8 Evaluation (`evaluation/eval_diarization.py`)

Both evaluations run on the 10 unseen speakers only. The formulas are in §11.
- **EER (speaker verification):**
  1. Take 20 clips per test speaker, 3 s each, and L2-normalise their embeddings.
  2. Score every pair (19,900 trials) by cosine similarity.
  3. Find the threshold where the false-accept rate equals the false-reject rate.
- **DER:** run on the 30 conversations, frame-level at 10 ms, with no forgiveness collar and Hungarian speaker mapping.

---

## 8. Module 3: Voice conversion

**Files:**
- `models/voice_conversion.py`
- `training/train_voice_conversion.py`
- `evaluation/eval_voice_conversion.py`
- `configs/voice_conversion.yaml`
- `third_party/openvoice/`

**The task:** keep **what** is said (the content: phonemes, timing, intonation) and change **who** says it (the timbre/identity).

$$
\text{converted} = \mathcal G\big(\text{content}(x_{\text{src}}),\ \text{identity}(x_{\text{tgt}})\big)
$$

Two implementations exist. They are selected by `backend` in `configs/voice_conversion.yaml`.

### 8.1 Custom backend: AutoVC + HiFi-GAN

**Paper:** Qian et al., *AutoVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss*, ICML 2019.

#### 8.1.1 Core idea: the information bottleneck

AutoVC is an **autoencoder** trained only to *reconstruct* its input, with the speaker embedding given separately:

$$
\hat X = D\big(E_c(X, \mathbf e_{\text{spk}}),\ \mathbf e_{\text{spk}}\big)
$$

The trick is that the content code $E_c(X)$ is made **too small to also carry the voice**. The decoder already receives the speaker identity $\mathbf e$ for free, so the cheapest way to reconstruct well is for the code to keep only what $\mathbf e$ doesn't provide, which is the content. The paper shows that with a correctly sized bottleneck, the code becomes (approximately) independent of the speaker.

At conversion time, the target speaker's embedding is swapped in:

$$
\hat X_{\text{converted}} = D\big(E_c(X_{\text{src}}, \mathbf e_{\text{src}}),\ \mathbf e_{\text{tgt}}\big)
$$

**How narrow is the bottleneck?** The code is `dim_neck = 16` channels per direction (32 numbers) every `freq = 16` frames. With a 16 ms hop, that is 32 numbers per 256 ms, or **125 numbers per second**. The 80-bin mel input is $80\times62.5 = 5000$ numbers per second, so the bottleneck is about 40× smaller.

#### 8.1.2 Input representation

- Mel spectrogram per §4.4: magnitude, 80 mels, $n_{\text{fft}}=1024$, hop 256, natural log, floor $10^{-5}$. This matches the vocoder.
- **Per-mel-bin standardisation:** $\tilde X = (X - \boldsymbol\mu)/\boldsymbol\sigma$, using the mean and std of each mel bin over all training frames. These are stored as buffers (`mel_mean`, `mel_std`) inside the checkpoint.
- **Speaker embedding:** the Module 2 embedding, L2-normalised (`speaker_embedding`). During training each speaker is represented by the **mean of up to 20 utterance embeddings**, re-normalised. That gives a stable speaker identity rather than a per-clip one.

#### 8.1.3 Architecture (7,436,608 parameters)

**Content encoder `E_c`:**

```
[mel (80) ; speaker emb (192) broadcast over time]  → 272 × T
  3 × [Conv1d(k=5) → BN → ReLU], 256 channels
  2-layer bidirectional LSTM, hidden = dim_neck = 16 per direction
  Downsample: forward state at frame 15, 31, 47 …  (end of each 16-frame block)
              backward state at frame 0, 16, 32 … (start of each block)
→ codes: (T/16) × 32
```

The forward LSTM has read the whole block once it reaches the block's *last* frame, and the backward LSTM has read it once it reaches the *first* frame. Sampling there means each code summarises its whole 16-frame block. The input is first padded with silence to a multiple of 16 frames (`pad_to_freq`).

**Decoder `D`:**

```
codes upsampled ×16 by repetition (repeat_interleave)        → T × 32
concatenate target speaker embedding (192) at every frame    → T × 224
LSTM (1 layer, 256)
3 × [Conv1d(k=5) → BN → ReLU], 256 ch
LSTM (2 layers, 512)
Linear 512 → 80                                              → mel_out (80 × T)
```

**Postnet:** 5 conv layers (k=5) with channels 80→256→256→256→256→80, BN on every layer and tanh on all but the last. It predicts a **residual correction**:

$$
X_{\text{post}} = X_{\text{out}} + \text{Postnet}(X_{\text{out}})
$$

#### 8.1.4 Losses (AutoVC paper, Eq. 6–8)

During training the source and target speaker are the **same** (`model(mel, emb, emb)`), which is the self-reconstruction setting.

$$
\mathcal L_{\text{recon}} = \lVert X_{\text{post}} - X\rVert_2^2 + \lVert X_{\text{out}} - X\rVert_2^2 \qquad(\text{mean squared error})
$$

$$
\mathcal L_{\text{content}} = \big\lVert E_c(X_{\text{post}}, \mathbf e) - E_c(X, \mathbf e)\big\rVert_1 \qquad(\text{mean absolute error})
$$

$$
\mathcal L = \lambda_r\,\mathcal L_{\text{recon}} + \lambda_c\,\mathcal L_{\text{content}},\qquad \lambda_r=\lambda_c=1
$$

The content-consistency loss says that re-encoding the reconstruction must give the same content code, which stabilises what the code represents.

#### 8.1.5 Training (`training/train_voice_conversion.py`)

1. **Precompute** the mels for every utterance of the 30 train speakers, plus each speaker's mean embedding. Audio is peak-normalised to 0.9 first.
2. **Compute normalisation statistics** (per-bin mean/std) from all training frames and write them into the model buffers.
3. **Each step:** take 16 random utterances, each with a random 128-frame (~2 s) crop, padded with the silence value if shorter.
4. **Optimiser:** Adam with lr $10^{-3}$, **cosine annealing** down to $10^{-5}$ over 8000 steps, and gradient clipping at 1.0.
5. **Validate** every 500 steps on 8 fixed batches of held-out utterances, and save the best checkpoint to `checkpoints/voice_conversion.pt`. The saved file also includes the per-speaker mean embeddings.
6. **Logged run:** the validation reconstruction MSE fell from 0.733 at step 500 to 0.260 at step 8000 (on normalised mels), and the content loss fell from 0.015 to about 0.003 (66 min of CPU training).

#### 8.1.6 Vocoder: HiFi-GAN (mel → waveform)

A mel spectrogram has no phase information and only 80 bands, so turning it back into audio needs a neural **vocoder**. The project uses the pretrained SpeechBrain `tts-hifigan-libritts-16kHz`, downloaded into `checkpoints/hifigan-libritts-16kHz/`.

**HiFi-GAN generator** (Kong et al., 2020), with this checkpoint's hyperparameters:

```
mel (80 × T)
Conv1d(80 → 512, k=7)
4 × [LeakyReLU(0.1) → ConvTranspose1d upsample → MRF]
      upsample factors 8, 8, 2, 2   (product = 256 = hop length, so 1 frame → 256 samples)
      channels 512 → 256 → 128 → 64 → 32
LeakyReLU → Conv1d(→1, k=7) → tanh   → waveform
```

**MRF (multi-receptive-field fusion):** after each upsampling step, three residual blocks with kernels 3, 7 and 11 (each with dilations 1, 3, 5) run in parallel and their outputs are averaged. This lets the generator model short and long periodic patterns at the same time.

HiFi-GAN was trained adversarially (with multi-period and multi-scale discriminators, plus mel and feature-matching losses). That training was done by SpeechBrain; this project only runs inference.

**Fallback:** if the vocoder can't be downloaded, the code uses **Griffin-Lim**. It iteratively estimates a phase consistent with the magnitude spectrogram:

$$
X^{(i+1)} = \text{STFT}\Big(\text{iSTFT}\big(|X|\,e^{j\angle X^{(i)}}\big)\Big)
$$

This is robotic-sounding but always available.

#### 8.1.7 `convert()` in custom mode

1. Get the source and target embeddings from Module 2 (L2-normalised).
2. Peak-normalise the source to 0.9, compute its mel and standardise it.
3. Run AutoVC with the *source* embedding into the encoder and the *target* embedding into the decoder. Take `mel_post`.
4. De-normalise, vocode with HiFi-GAN, crop to the source length and peak-normalise to 0.9.

**Shape example**, a 4 s source:
- 64,000 samples → $64000/256+1 = 251$ mel frames.
- Padded to 256 → codes of 16 × 32.
- Upsampled back to 256 frames → mel of 80 × 256, cropped to 251.
- HiFi-GAN → about 64,256 samples → cropped to 64,000.

### 8.2 Pretrained backend (current default): OpenVoice v2 tone-colour converter

The converter network is vendored from `github.com/myshell-ai/OpenVoice` (MIT licence) in `third_party/openvoice/`. Its weights are in `checkpoints/pretrained/openvoice_v2/converter/`. Only the converter is used; OpenVoice's TTS and text front-end are not. It has 32,792,226 parameters and runs at **22,050 Hz**: audio is resampled in and back out.

It is built on **VITS** (Kim et al., 2021): a variational autoencoder whose latent passes through a **normalising flow**, followed by a HiFi-GAN-style decoder.

#### 8.2.1 Components

| Component | What it does | Details |
|---|---|---|
| **Linear spectrogram** | input features | STFT magnitude, $n_{\text{fft}}=1024$ → **513 bins**, hop 256, Hann window. No mel compression. |
| **Reference (tone-colour) encoder** | extracts a voice "colour" vector $\mathbf g\in\mathbb R^{256}$ | LayerNorm over the 513 bins, then 6 × Conv2d(3×3, stride 2) with channels 32, 32, 64, 64, 128, 128 (frequency axis 513 → 9). Each frame becomes 128 × 9 = 1152 features → GRU(128) → final hidden state → Linear(128 → 256). |
| **Posterior encoder** $q(z\mid y)$ | spectrogram → latent $z$ (192 channels) | 1×1 conv → 16-layer WaveNet-style stack (WN, kernel 5) → mean $\mathbf m$ and log-std $\log\boldsymbol\sigma$ |
| **Flow** $f_\theta(\cdot\,;\mathbf g)$ | invertible map, conditioned on the voice | 4 residual affine-coupling layers (mean-only) with channel flips between them |
| **Generator** | latent → waveform | HiFi-GAN: upsample 8 × 8 × 2 × 2 = 256, MRF kernels 3/7/11 |

`zero_g = true` in the config means the **posterior encoder and the generator receive zeros** instead of the speaker vector. **Only the flow sees the speaker.** So the flow alone is responsible for removing the source timbre and adding the target's.

#### 8.2.2 The maths of conversion

**Speaker vectors.** `speaker_embedding` first removes silent frames using Module 2's VAD (`_speech_only`). It then splits the speech into chunks of at most 10 s and averages the reference-encoder output over them:

$$
\mathbf g_{\text{src}} = \text{RefEnc}(x_{\text{src}}),\qquad \mathbf g_{\text{tgt}} = \text{RefEnc}(x_{\text{tgt}})
$$

**Step 1: encode with controlled randomness** (the reparameterisation trick):

$$
\mathbf z = \mathbf m + \boldsymbol\epsilon\odot\tau\,\exp(\log\boldsymbol\sigma),\qquad \boldsymbol\epsilon\sim\mathcal N(0, I)
$$

$\tau$ = `tau` = 0.3 (from the config) scales the noise. A lower $\tau$ is more deterministic, which the config comment describes as "closer to target timbre".

**Step 2: the flow strips the source voice.** Each coupling layer splits the channels into halves $[\mathbf z_0, \mathbf z_1]$ and shifts one half by an amount computed from the other half and the speaker vector:

$$
\text{forward: }\ \mathbf z_1' = \mathbf z_1 + \mu_\phi(\mathbf z_0, \mathbf g),\qquad
\text{inverse: }\ \mathbf z_1 = \mathbf z_1' - \mu_\phi(\mathbf z_0, \mathbf g)
$$

The **Flip** layers reverse the channel order between couplings, so both halves get transformed. Because the scale is fixed to 1 ("mean-only"), every layer is volume-preserving: its log-determinant is 0. Then

$$
\mathbf z_p = f_\theta(\mathbf z;\ \mathbf g_{\text{src}})
$$

During VITS training, $\mathbf z_p$ is pulled towards a speaker-independent prior, so it represents mostly *content*.

**Step 3: the inverse flow adds the target voice:**

$$
\hat{\mathbf z} = f_\theta^{-1}(\mathbf z_p;\ \mathbf g_{\text{tgt}})
$$

**Step 4: decode** $\hat{\mathbf z}$ to a 22.05 kHz waveform with the HiFi-GAN generator. Then resample to 16 kHz, pad or crop to the source length, and peak-normalise to 0.9.

**WaveNet (WN) gated unit**, used inside the posterior encoder and the flow couplings. Each layer is:

$$
\mathbf a = \tanh(\mathbf W_f * \mathbf x + \mathbf V_f\,\mathbf g)\ \odot\ \sigma(\mathbf W_g * \mathbf x + \mathbf V_g\,\mathbf g)
$$

plus residual and skip 1×1 convolutions. The tanh half proposes a value, and the sigmoid "gate" decides how much of it passes. The speaker vector $\mathbf g$ conditions every layer through $\mathbf V\mathbf g$.

**Why this works zero-shot:** the reference encoder was trained on thousands of voices. Any new voice therefore maps to a meaningful $\mathbf g$, and a few seconds of reference audio are enough. AutoVC trained on only 30 speakers, by contrast, generalises poorly to unseen voices (see the results).

---

## 9. Pitch modulation (DSP)

**Function:** `modulate(audio, sr, pitch_factor)` in `models/voice_conversion.py`. It is pure signal processing with no learned parameters.

A shift of $n$ semitones multiplies every frequency by

$$
r = 2^{n/12}
$$

For example, +12 semitones is one octave up ($r=2$), +3 gives $r\approx1.19$, and −12 gives $r=0.5$. The app's slider covers −12…+12.

**Implementation:** the code first tries **pyrubberband**, which needs the Rubber Band command-line tool. If that isn't available it falls back to `librosa.effects.pitch_shift`, which works in two steps:

1. **Time-stretch** by a factor of $1/r$ with a **phase vocoder**. In the STFT domain, frames are re-spaced, and each bin's phase is advanced by its measured instantaneous frequency:
   $$\phi_{t+1}[k] = \phi_t[k] + \Delta\phi[k]$$
   where $\Delta\phi$ is estimated from consecutive input frames. This keeps the sinusoids continuous, so the duration changes while the pitch stays the same.
2. **Resample** by $r$. This changes the pitch and restores the original duration.

The result has the same length, with the pitch multiplied by $r$.

---

## 10. Optimisers and learning-rate schedules

**Adam** (all three trainings). For each parameter $\theta$ with gradient $g_t$ (Adam defaults $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$):

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,\qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$

$$
\hat m_t = \frac{m_t}{1-\beta_1^t},\quad \hat v_t = \frac{v_t}{1-\beta_2^t},\qquad
\theta_t = \theta_{t-1} - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

With weight decay $\lambda$ (Module 2), $\lambda\theta$ is added to $g_t$ (L2 regularisation).

**Gradient-norm clipping** (Modules 1 and 3): if $\lVert\mathbf g\rVert_2 > c$, rescale $\mathbf g \leftarrow \mathbf g\cdot c/\lVert\mathbf g\rVert_2$. Here $c$ = 5 for separation and 1 for AutoVC. This prevents exploding updates, especially in LSTMs.

| Module | Schedule | Formula / rule |
|---|---|---|
| 1 Separation | `ReduceLROnPlateau` (factor 0.5, patience 2) | if validation loss hasn't improved for 2 checks, $\eta\leftarrow\eta/2$ |
| 2 ECAPA | `OneCycleLR` (peak $2\times10^{-3}$, 15% warm-up) | warm-up: cosine ramp from $\eta_{\max}/25$ to $\eta_{\max}$ over the first 15% of steps; then cosine decay down to $\eta_{\max}/(25\cdot10^4)$ |
| 3 AutoVC | `CosineAnnealingLR` ($T=8000$, $\eta_{\min}=10^{-5}$) | $\eta_t = \eta_{\min} + \tfrac12(\eta_{\max}-\eta_{\min})\big(1+\cos\frac{\pi t}{T}\big)$ |

**Seeding:** `set_seed(42)` seeds Python's `random`, NumPy and PyTorch (CPU and CUDA), so runs are repeatable.

**Checkpoint selection:** every training script keeps the weights with the **best validation score**, not the final weights.

---

## 11. Evaluation metrics

### 11.1 SI-SDRi (separation)

See §6.6 and §6.9. Higher is better; 0 dB means no improvement over the mixture. The published Conv-TasNet result on WSJ0-2mix is about 15 dB.

### 11.2 EER: equal error rate (speaker verification)

For a threshold $\theta$, each trial (a pair of clips) is accepted as "same speaker" if $\cos > \theta$. Then:

$$
\text{FPR}(\theta) = \frac{\#\{\text{different-speaker pairs accepted}\}}{\#\{\text{different-speaker pairs}\}},\qquad
\text{FNR}(\theta) = \frac{\#\{\text{same-speaker pairs rejected}\}}{\#\{\text{same-speaker pairs}\}}
$$

Raising $\theta$ lowers FPR and raises FNR. The **EER** is the error rate at the threshold where they are equal, $\text{FPR}=\text{FNR}$. The code reads it from the ROC curve, taking the point where $|\text{FNR}-\text{FPR}|$ is smallest and averaging the two. Lower is better.

### 11.3 DER: diarization error rate

At 10 ms frame level, over the reference speech time:

$$
\text{DER} = \frac{T_{\text{miss}} + T_{\text{false alarm}} + T_{\text{confusion}}}{T_{\text{reference speech}}}
$$

- **Miss:** reference says speech, prediction says silence.
- **False alarm:** reference says silence, prediction says speech.
- **Confusion:** both say speech, but the (mapped) speakers differ.

The predicted cluster names ("Speaker_0") are arbitrary, so they are first mapped to the true speakers with the **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`). It finds the one-to-one mapping that maximises total overlap:

$$
\pi^* = \arg\max_{\pi}\sum_{h}\big|\{t : \text{hyp}_t = h,\ \text{ref}_t = \pi(h)\}\big|
$$

The overall DER is the **duration-weighted** average over the 30 files. No forgiveness collar is used around boundaries, which makes this stricter than the standard NIST scoring.

### 11.4 Voice-conversion speaker similarity

The metric uses the cosine similarity of L2-normalised Module 2 embeddings. The target score uses a *different* target utterance from the one used for conditioning. For each conversion it reports:

| Metric | Meaning | Want |
|---|---|---|
| sim(converted, target) | sounds like the target? | high |
| sim(converted, source) | still sounds like the source? | low |
| sim(source, target) | baseline: how alike the two speakers already were | — |
| success | sim(conv, target) > sim(conv, source) | close to 100% |

It is reported for **seen** speakers (train speakers, held-out utterances) and **unseen** speakers (the 10 test speakers, i.e. zero-shot). Caveat: the same ECAPA model both conditions AutoVC and scores it, which flatters the result. The script also prints a **MOS** (mean opinion score, 1–5 naturalness) survey template for human listening tests.

---

## 12. The Streamlit app

**File:** `app.py`. Run it with `streamlit run app.py`.

1. **Sidebar input.** Choose a file in `samples/` or upload a wav/flac/mp3. The input is loaded as 16 kHz mono. Changing the input clears all downstream results stored in `st.session_state`.
2. **Step 1: Separate Speakers.** Calls `separate_waveform(mixture)` and shows the waveform, the mel spectrogram (80 mels, dB) and audio players for both streams.
3. **Step 2: Diarization**, in two tabs:
   - **Separated streams:** `get_embedding` on each stream and their cosine similarity. A value above 0.6 triggers a warning that the streams sound like the same speaker, so separation may have failed. It also shows a speech-activity timeline per stream from the VAD. After separation, "who spoke when" is simply each stream's activity.
   - **Diarize input:** runs `diarize()` on the original audio with a chosen number of speakers (1–6), and shows a timeline plus the segment list. Use `sample_conversation_0000.wav`, which has turn-taking speech.
4. **Step 3: Convert Voice.** Pick a source stream. The target reference is either an uploaded clip or, by default, the other stream. The app calls `convert()`, then shows three similarity metrics: converted→target (with its change versus the source↔target baseline), converted→source, and source↔target.
5. **Pitch Modulation.** Pick the converted output, stream 1 or stream 2, and shift it by −12…+12 semitones with `modulate()`.

Errors are shown in the UI rather than crashing the app (`run_step`). Models load lazily on first use and are cached in module-level globals (`_model`).

---

## 13. Results

These come from `results/metrics/*.csv` and `logs/`. **They were measured with the custom models.** The run happened before the configs were switched to the pretrained backends.

| Module | Model | Training | Test result (unseen speakers) |
|---|---|---|---|
| Separation | Conv-TasNet, 2.5 M params | 4000 steps × batch 4, 81 min CPU | **SI-SDRi 3.72 dB** mean / 3.81 dB median (200 mixtures; ≈82% improved) |
| Diarization | ECAPA-TDNN, 1.7 M params, AAM-Softmax | ≈20 epochs, ≈13 min | **EER 10.9%** (3 s clips; mean cosine same-speaker 0.649 vs different-speaker 0.097). **DER 14.4%** = miss 1.5% + false alarm 3.8% + confusion 9.1%. Closed-set validation accuracy 100%. |
| Voice conversion | AutoVC + HiFi-GAN | 8000 steps × batch 16, 66 min | **Seen speakers:** sim(conv, target) **0.62** vs source↔target baseline 0.12; 95% success. **Unseen speakers:** **0.30**, 70% success. |

**How to read these:**
- **Separation is under-trained.** Validation SI-SDR was still rising when training stopped. More `training.steps` would help. The pretrained Asteroid model is far stronger: a code comment notes about 15 dB SI-SDRi on this test set.
- **Diarization confusion (9.1%)** dominates the DER. That means VAD works well, and the errors come from embeddings of unseen speakers occasionally clustering wrongly.
- **AutoVC on unseen speakers is weak**, which is expected with only 30 training voices. This is the main reason the default backend is OpenVoice v2.

Plots:
- `results/loss_curves/`: training curves for all three models.
- `results/metrics/tsne_embeddings.png`: the embedding clusters of the unseen speakers.

Audio: `results/audio_examples/`.

---

## 14. Tests

There are 39 pytest tests in `tests/`, run with `python -m pytest tests -q`. All 39 pass.

| File | What it checks |
|---|---|
| `test_separation.py` | Conv-TasNet builds; output shape is (batch, 2, T); output length equals input length, even for odd lengths; SI-SDR is high for a perfect signal, very low for orthogonal signals, and scale-invariant; PIT gives the same loss for swapped outputs; `separate()` returns two arrays |
| `test_diarization.py` | ECAPA builds; embedding shape is (B, 192); no leftover classifier head; AAM loss is lower when embeddings align with their class; the cosine prediction is correct; log-mel shape; DER is 0 for a perfect prediction, label names don't matter, and one cluster for two speakers gives confusion; `get_embedding` shape and determinism; `diarize` returns non-overlapping tuples |
| `test_voice_conversion.py` | AutoVC shapes; output frames equal input frames for many lengths; embedding-dim assertion; bottleneck shape and narrowness; mel shape; `modulate` with 0, +/− shifts; `convert` really calls Module 2's `get_embedding` (cross-module integration); converted length equals source length |

Tests that need trained checkpoints are skipped automatically if those checkpoints are missing.

---

## 15. How to run everything

```bash
# 1. Install (Python 3.10)
pip install -r requirements.txt
pip install asteroid            # needed by the default separation backend (missing from requirements.txt)

# 2. Data: downloads LibriSpeech dev-clean if missing, builds the split and test sets
python scripts/prepare_data.py

# 3. Train + evaluate all custom models (~3 h on CPU), then run the tests
python scripts/train_all.py

#    …or individually:
python training/train_separation.py --config configs/separation.yaml
python evaluation/eval_separation.py
python training/train_diarization.py --config configs/diarization.yaml
python training/train_diarization.py --config configs/diarization.yaml --tsne
python evaluation/eval_diarization.py
python training/train_voice_conversion.py --config configs/voice_conversion.yaml
python evaluation/eval_voice_conversion.py

# 4. Demo
streamlit run app.py

# 5. Tests
python -m pytest tests -q
```

The training scripts accept `--threads N` to limit CPU threads. `train_diarization.py` also accepts `--workers N` for the data-loader workers.

**Order matters for custom voice conversion:** `train_voice_conversion.py` uses Module 2's embeddings, so train diarization first.

---

## 16. Known issues and limitations

1. **`asteroid` is missing from `requirements.txt`.** The default (pretrained) separation backend imports it, so a fresh install fails at "Separate" until you `pip install asteroid`.
2. **The README results describe the custom models.** The app now uses the pretrained backends by default. The README sentence "Module 3 uses Module 2's trained, frozen embedding network" is only true with `backend: custom` for voice conversion.
3. **Embedding mismatch risk.** AutoVC was trained on embeddings from whichever diarization backend was active at the time (custom ECAPA). If you set only voice conversion to `custom` while diarization stays `pretrained`, AutoVC will receive SpeechBrain embeddings it never saw, and quality will drop. Set **both** to `custom` together.
4. **No overlap handling in diarization.** `diarize()` assumes one speaker at a time. For overlapping speech, use separation first and then per-stream activity (the app's first Step 2 tab).
5. **Two speakers only.** Separation is fixed at `num_sources: 2`.
6. **The VC similarity score is biased**, because the same embedding model conditions and scores it. Human listening (MOS) is the honest check.
7. **Small data.** 30 training speakers and about 4 h of audio limit every custom model, especially zero-shot voice conversion.
8. The Asteroid model's `sample_rate` attribute says 8000. According to the code comment this is a stale default in its uploaded config: the model was trained at 16 kHz and is used at 16 kHz.
9. `logs/pipeline.log` ends with `ALL DONE rc=1`, so the last step of that ad-hoc run failed. The tests currently pass.

---

## 17. Glossary

| Term | Meaning |
|---|---|
| **Mixture** | one signal containing several voices added together |
| **Mask** | per-time, per-feature weights saying how much of the mixture belongs to one source |
| **TCN** | temporal convolutional network: stacked dilated 1-D convolutions |
| **gLN** | global layer normalisation over channels and time |
| **SI-SDR / SI-SDRi** | scale-invariant signal-to-distortion ratio / its improvement over the mixture |
| **PIT / uPIT** | (utterance-level) permutation-invariant training: score the best output-to-target assignment |
| **Embedding** | fixed-length vector representing a speaker's voice (192-dim here) |
| **TDNN** | time-delay neural network, i.e. 1-D convolution over time |
| **SE** | squeeze-and-excitation channel attention |
| **Res2Net** | block that splits channels into groups for multi-scale processing |
| **ASP** | attentive statistics pooling: attention-weighted mean and std over time |
| **AAM-Softmax** | additive angular margin softmax (ArcFace) loss |
| **VAD** | voice activity detection (speech vs non-speech) |
| **Diarization** | "who spoke when" segmentation of a recording |
| **EER** | equal error rate, where false accept = false reject |
| **DER** | diarization error rate = miss + false alarm + confusion |
| **Bottleneck** | deliberately small representation that forces information to be discarded |
| **Vocoder** | model that turns a spectrogram back into a waveform |
| **HiFi-GAN** | GAN-trained neural vocoder with multi-receptive-field fusion |
| **Normalising flow** | an invertible neural network; its exact inverse is used to swap speakers in OpenVoice |
| **Tone colour** | OpenVoice's term for timbre (voice identity) |
| **Semitone** | 1/12 of an octave; frequency ratio $2^{1/12}\approx1.0595$ |
| **Phase vocoder** | STFT-based method for time-stretching without changing pitch |
| **MOS** | mean opinion score, human 1–5 naturalness rating |

---

## 18. References

1. Y. Luo, N. Mesgarani. *Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation.* IEEE/ACM TASLP 2019. arXiv:1809.07454
2. D. Yu, M. Kolbæk, Z.-H. Tan, J. Jensen. *Permutation Invariant Training of Deep Models for Speaker-Independent Multi-talker Speech Separation.* ICASSP 2017.
3. J. Le Roux et al. *SDR — Half-baked or Well Done?* (SI-SDR). ICASSP 2019.
4. B. Desplanques, J. Thienpondt, K. Demuynck. *ECAPA-TDNN.* Interspeech 2020. arXiv:2005.07143
5. J. Deng et al. *ArcFace: Additive Angular Margin Loss for Deep Face Recognition.* CVPR 2019.
6. J. Hu et al. *Squeeze-and-Excitation Networks.* CVPR 2018. S.-H. Gao et al. *Res2Net.* TPAMI 2019.
7. K. Qian et al. *AutoVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss.* ICML 2019. arXiv:1905.05879
8. J. Kong, J. Kim, J. Bae. *HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis.* NeurIPS 2020.
9. J. Kim, J. Kong, J. Son. *VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech.* ICML 2021.
10. Z. Qin et al. *OpenVoice: Versatile Instant Voice Cloning.* 2023. github.com/myshell-ai/OpenVoice
11. V. Panayotov et al. *LibriSpeech: an ASR corpus based on public domain audio books.* ICASSP 2015.
12. Pretrained weights: `JorisCos/ConvTasNet_Libri2Mix_sepclean_16k`, `speechbrain/spkrec-ecapa-voxceleb`, `speechbrain/tts-hifigan-libritts-16kHz`, `myshell-ai/OpenVoiceV2` (all on Hugging Face).
