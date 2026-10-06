"""
Module 2 — Speaker Embedding & Diarization (ECAPA-TDNN)

Maps a short audio clip to a fixed-length embedding such that same-speaker clips
cluster together in embedding space. Provides diarization ("who spoke when").

Architecture: ECAPA-TDNN
  - Frame-level TDNN/1D-conv layers over log-mel spectrograms
  - Squeeze-Excitation (SE) and Res2Net blocks
  - Attentive statistics pooling to utterance-level vector
  - Fully-connected embedding layer (192-dim output)

The trained embedding model is reused by Module 3 (voice conversion) as its
speaker-identity encoder — this is the cross-module integration point.

Reference: Desplanques et al., "ECAPA-TDNN," arxiv.org/abs/2005.07143
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import librosa

# ─── Canonical pipeline sample rate ───
SAMPLE_RATE = 16000
EMBEDDING_DIM = 192  # Must match voice_conversion.yaml


# ═══════════════════════════════════════════════════════════
# Feature Extraction
# ═══════════════════════════════════════════════════════════

def extract_log_mel(audio, sr, n_mels=80, n_fft=512, hop_length=160, win_length=400):
    """Extract log-mel spectrogram features from audio waveform."""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, n_mels=n_mels
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # Per-utterance mean normalization (CMN) removes channel/loudness offsets
    log_mel = log_mel - log_mel.mean(axis=1, keepdims=True)
    return log_mel.astype(np.float32)  # (n_mels, frames)


# ═══════════════════════════════════════════════════════════
# Model Components
# ═══════════════════════════════════════════════════════════

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel recalibration."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, channels, frames)
        b, c, _ = x.shape
        se = self.squeeze(x).view(b, c)
        se = self.excitation(se).view(b, c, 1)
        return x * se


class Res2NetBlock(nn.Module):
    """Res2Net-style multi-scale feature block."""

    def __init__(self, channels, kernel_size=3, dilation=1, scale=8):
        super().__init__()
        self.scale = scale
        assert channels % scale == 0
        width = channels // scale

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(scale - 1):
            self.convs.append(nn.Conv1d(width, width, kernel_size,
                                        dilation=dilation,
                                        padding=(kernel_size - 1) * dilation // 2))
            self.bns.append(nn.BatchNorm1d(width))

    def forward(self, x):
        # x: (batch, channels, frames)
        chunks = torch.chunk(x, self.scale, dim=1)
        outputs = [chunks[0]]
        for i in range(1, self.scale):
            if i == 1:
                y = chunks[i]
            else:
                y = chunks[i] + outputs[-1]
            y = self.bns[i - 1](F.relu(self.convs[i - 1](y)))
            outputs.append(y)
        return torch.cat(outputs, dim=1)


class SERes2NetBlock(nn.Module):
    """ECAPA-TDNN SE-Res2Net block: 1D conv → Res2Net → 1D conv → SE + residual."""

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, scale=8):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.res2net = Res2NetBlock(out_channels, kernel_size, dilation, scale)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.res2net(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.se(out)
        return out + residual


class AttentiveStatisticsPooling(nn.Module):
    """Attentive statistics pooling: attention-weighted mean + std."""

    def __init__(self, channels, attention_channels=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, attention_channels, 1),
            nn.ReLU(),
            nn.Conv1d(attention_channels, channels, 1),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # x: (batch, channels, frames)
        attn = self.attention(x)  # (batch, channels, frames)
        mean = torch.sum(x * attn, dim=-1)  # (batch, channels)
        # Weighted std
        var = torch.sum(attn * (x - mean.unsqueeze(-1)) ** 2, dim=-1)
        std = torch.sqrt(var.clamp(min=1e-8))
        return torch.cat([mean, std], dim=1)  # (batch, 2*channels)


class ECAPATDNN(nn.Module):
    """ECAPA-TDNN speaker embedding network.

    Trained with AAM-Softmax (whose class-weight matrix lives in AAMSoftmax and is
    discarded after training); the embedding-layer output is used for new speakers.
    """

    def __init__(self, n_mels=80, channels=512, embedding_dim=192, attention_channels=128):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Initial TDNN layer
        self.conv1 = nn.Conv1d(n_mels, channels, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(channels)

        # SE-Res2Net blocks with increasing dilation
        self.layer1 = SERes2NetBlock(channels, channels, kernel_size=3, dilation=2)
        self.layer2 = SERes2NetBlock(channels, channels, kernel_size=3, dilation=3)
        self.layer3 = SERes2NetBlock(channels, channels, kernel_size=3, dilation=4)

        # Multi-layer feature aggregation (MFA)
        self.mfa_conv = nn.Conv1d(channels * 3, channels * 3, 1)
        self.mfa_bn = nn.BatchNorm1d(channels * 3)

        # Attentive statistics pooling
        self.asp = AttentiveStatisticsPooling(channels * 3, attention_channels)

        # Final embedding layer
        self.fc = nn.Linear(channels * 3 * 2, embedding_dim)
        self.bn_emb = nn.BatchNorm1d(embedding_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, n_mels, frames) log-mel spectrogram

        Returns:
            (batch, embedding_dim) speaker embedding
        """
        out = F.relu(self.bn1(self.conv1(x)))

        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)

        # Multi-layer feature aggregation
        mfa = torch.cat([out1, out2, out3], dim=1)
        mfa = F.relu(self.mfa_bn(self.mfa_conv(mfa)))

        # Pooling
        pooled = self.asp(mfa)

        # Embedding
        return self.bn_emb(self.fc(pooled))


# ═══════════════════════════════════════════════════════════
# Loss: AAM-Softmax (Additive Angular Margin Softmax / ArcFace)
# ═══════════════════════════════════════════════════════════

class AAMSoftmax(nn.Module):
    """Additive Angular Margin Softmax (ArcFace-style) loss.

    Maximizes angular separation between speaker classes in embedding space.
    """

    def __init__(self, embedding_dim, num_classes, margin=0.2, scale=30):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.ce = nn.CrossEntropyLoss()

    def cosine(self, embeddings):
        """(batch, num_classes) cosine similarity to each class centre.
        argmax of this is the classifier's prediction (used for accuracy)."""
        return F.linear(F.normalize(embeddings, dim=1), F.normalize(self.weight, dim=1))

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (batch, embedding_dim) normalized embeddings
            labels: (batch,) speaker class indices

        Returns:
            loss: scalar AAM-Softmax loss
        """
        cosine = self.cosine(embeddings)  # (batch, num_classes)

        # Add angular margin to the target class
        theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
        one_hot = F.one_hot(labels, cosine.shape[1]).float()
        target_cosine = torch.cos(theta + self.margin * one_hot)

        # Scale and compute loss
        logits = self.scale * target_cosine
        return self.ce(logits, labels)


# ═══════════════════════════════════════════════════════════
# Public API — Interface Contract (Section 5)
# ═══════════════════════════════════════════════════════════

_model = None
_config = None
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checkpoints')


def load_config():
    global _config
    if _config is None:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'diarization.yaml')
        with open(config_path, 'r') as f:
            _config = yaml.safe_load(f)
    return _config


def build_model(config):
    return ECAPATDNN(
        n_mels=config['data'].get('n_mels', 80),
        channels=config['model'].get('channels', 512),
        embedding_dim=config['model'].get('embedding_dim', EMBEDDING_DIM),
        attention_channels=config['model'].get('attention_channels', 128),
    )


def backend():
    return load_config().get('backend', 'pretrained')


def _load_model(checkpoint_path=None):
    """Load the embedding model selected by `backend` in configs/diarization.yaml.

    - pretrained (default): SpeechBrain ECAPA-TDNN trained on VoxCeleb 1+2 (~7k speakers),
      huggingface.co/speechbrain/spkrec-ecapa-voxceleb. Same architecture and the same
      192-dim embedding as ours.
    - custom: our ECAPA-TDNN trained by training/train_diarization.py (30 speakers).
    """
    config = load_config()
    if backend() == 'pretrained':
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy
        source = config.get('pretrained', {}).get('source', 'speechbrain/spkrec-ecapa-voxceleb')
        model = EncoderClassifier.from_hparams(
            source=source, savedir=os.path.join(CHECKPOINT_DIR, 'pretrained', source.split('/')[-1]),
            local_strategy=LocalStrategy.COPY)
        model.eval()
        print(f"[Diarization] Loaded pretrained {source}")
        return model

    model = build_model(config)

    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, 'diarization.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No diarization checkpoint at {checkpoint_path}. Train it with "
            f"`python training/train_diarization.py` first.")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state)
    print(f"[Diarization] Loaded checkpoint from {checkpoint_path}")

    model.eval()
    return model


def _get_model():
    """Get or lazily load the cached model."""
    global _model
    if _model is None:
        _model = _load_model()
    return _model


def features(audio):
    """Log-mel features with the parameters from configs/diarization.yaml."""
    d = load_config()['data']
    return extract_log_mel(audio, SAMPLE_RATE, n_mels=d.get('n_mels', 80), n_fft=d.get('n_fft', 512),
                           hop_length=d.get('hop_length', 160), win_length=d.get('win_length', 400))


def embed_batch(segments) -> np.ndarray:
    """Embeddings for a list of equal-length 16 kHz waveforms -> (n, 192)."""
    model = _get_model()
    with torch.no_grad():
        if backend() == 'pretrained':
            wavs = torch.from_numpy(np.stack(segments).astype(np.float32))
            return model.encode_batch(wavs)[:, 0].cpu().numpy()
        feats = torch.from_numpy(np.stack([features(s) for s in segments]))
        return model(feats).cpu().numpy()


def get_embedding(audio: np.ndarray, sr: int) -> np.ndarray:
    """Returns a fixed-length speaker embedding vector (192-dim).

    Args:
        audio: Audio waveform as numpy array.
        sr: Sample rate of the audio.

    Returns:
        192-dimensional speaker embedding as numpy array (not length-normalized;
        compare embeddings with cosine similarity).
    """
    audio = np.asarray(audio, dtype=np.float32)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return embed_batch([audio])[0]


def speech_activity(audio, sr, frame_sec=0.01, threshold_db=35.0):
    """Energy-based voice activity detection.

    Returns a boolean array with one entry per `frame_sec` frame: True where the frame
    energy is within `threshold_db` of the loud (95th percentile) part of the signal.
    """
    hop = int(frame_sec * sr)
    rms = librosa.feature.rms(y=audio, frame_length=hop * 4, hop_length=hop, center=True)[0]
    db = 20 * np.log10(rms + 1e-8)
    return db > (np.percentile(db, 95) - threshold_db)


def diarize(audio: np.ndarray, sr: int, num_speakers: int = None) -> list:
    """Performs speaker diarization: "who spoke when."

    1. Energy VAD marks speech frames (10 ms).
    2. Sliding windows over the audio; windows that are mostly speech get an embedding.
    3. Length-normalized embeddings are clustered (agglomerative, cosine/average linkage).
    4. Each speech frame takes the label of the nearest window centre, consecutive
       frames with the same label are merged, and short gaps are bridged.

    Assumes non-overlapping speech (one speaker at a time).

    Args:
        audio: Audio waveform as numpy array.
        sr: Sample rate of the audio.
        num_speakers: Number of speakers (default: config diarization.num_clusters).

    Returns:
        List of (speaker_id, start_sec, end_sec) tuples, sorted by start time.
    """
    from sklearn.cluster import AgglomerativeClustering

    cfg = load_config().get('diarization', {})
    window_sec = cfg.get('window_sec', 1.5)
    hop_sec = cfg.get('hop_sec', 0.25)
    min_gap_sec = cfg.get('min_gap_sec', 0.5)
    num_speakers = num_speakers or cfg.get('num_clusters', 2)

    audio = np.asarray(audio, dtype=np.float32)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    frame_sec = 0.01
    vad = speech_activity(audio, sr, frame_sec)
    n_frames = len(vad)
    if not vad.any():
        return []

    # Sliding windows (last one aligned to the end of the clip)
    win, hop = int(window_sec * sr), int(hop_sec * sr)
    starts = list(range(0, max(len(audio) - win, 0) + 1, hop))
    if len(audio) > win and starts[-1] != len(audio) - win:
        starts.append(len(audio) - win)

    kept_starts, segs = [], []
    for st in starts:
        f0 = int(st / sr / frame_sec)
        f1 = int(min(st + win, len(audio)) / sr / frame_sec)
        if vad[f0:f1].mean() >= 0.5:
            seg = audio[st:st + win]
            if len(seg) < win:
                seg = np.pad(seg, (0, win - len(seg)))
            kept_starts.append(st)
            segs.append(seg)
    if not segs:
        return []

    embs = np.concatenate([embed_batch(segs[i:i + 64]) for i in range(0, len(segs), 64)])
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)

    if len(embs) < num_speakers:
        labels = np.arange(len(embs))
    else:
        labels = AgglomerativeClustering(n_clusters=num_speakers, metric='cosine',
                                         linkage='average').fit_predict(embs)

    # Frame labelling: nearest window centre, speech frames only
    centres = (np.array(kept_starts) + win / 2) / sr
    frame_times = (np.arange(n_frames) + 0.5) * frame_sec
    nearest = np.abs(frame_times[:, None] - centres[None, :]).argmin(axis=1)
    frame_labels = np.where(vad, labels[nearest], -1)

    # Bridge short non-speech gaps inside the same speaker's turn
    max_gap = int(min_gap_sec / frame_sec)
    i = 0
    while i < n_frames:
        if frame_labels[i] != -1:
            i += 1
            continue
        j = i
        while j < n_frames and frame_labels[j] == -1:
            j += 1
        if i > 0 and j < n_frames and j - i <= max_gap and frame_labels[i - 1] == frame_labels[j]:
            frame_labels[i:j] = frame_labels[j]
        i = j

    # Frames -> segments
    segments = []
    i = 0
    while i < n_frames:
        lab = frame_labels[i]
        j = i
        while j < n_frames and frame_labels[j] == lab:
            j += 1
        if lab != -1 and (j - i) * frame_sec >= 0.1:
            segments.append((f"Speaker_{int(lab)}", round(i * frame_sec, 2),
                             round(min(j * frame_sec, len(audio) / sr), 2)))
        i = j
    return segments
