"""
Module 3 — Voice Conversion & Modulation (AutoVC-style + HiFi-GAN)

Converts speech from a source speaker into a target speaker's voice while
preserving linguistic content. Applies controllable pitch modulation on top.

Architecture: Scaled-down AutoVC (Qian et al., 2019)
  - Content encoder: conv stack + BLSTM, downsampled by `freq` (the information
    bottleneck that forces disentanglement of content from identity)
  - Speaker encoder: REUSES Module 2's trained, frozen ECAPA-TDNN (not a separate model)
  - Decoder: content codes are upsampled back to the input frame rate, conditioned on
    the target speaker embedding, and decoded to a mel-spectrogram (+ postnet)
  - Vocoder: pretrained 16 kHz HiFi-GAN (SpeechBrain, LibriTTS) converts mel to waveform

Cross-module integration: The speaker encoder is Module 2's trained model (frozen).

Reference: Qian et al., "AutoVC," arxiv.org/abs/1905.05879
"""

import os

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# Import Module 2's embedding function — the real cross-module integration
from models.diarization import EMBEDDING_DIM, get_embedding

# ─── Canonical pipeline sample rate ───
SAMPLE_RATE = 16000
N_MELS = 80
LOG_MEL_FLOOR = float(np.log(1e-5))  # value of a silent mel bin

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')


# ═══════════════════════════════════════════════════════════
# Mel Spectrogram Utilities
# ═══════════════════════════════════════════════════════════

def audio_to_mel(audio, sr=SAMPLE_RATE, n_fft=1024, hop_length=256, win_length=1024,
                 n_mels=N_MELS, fmin=0, fmax=8000):
    """Log magnitude mel-spectrogram matching the pretrained HiFi-GAN vocoder
    (speechbrain/tts-hifigan-libritts-16kHz: power=1, slaney mel, natural log, 1e-5 floor).

    Mismatched parameters produce audible noise, not degraded speech.
    """
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, n_mels=n_mels, fmin=fmin, fmax=fmax, power=1.0
    )
    return np.log(np.clip(mel, a_min=1e-5, a_max=None)).astype(np.float32)  # (n_mels, frames)


def speaker_embedding(audio, sr=SAMPLE_RATE):
    """Module 2 embedding, L2-normalized (AutoVC conditions on a unit-norm vector)."""
    emb = get_embedding(audio, sr)
    return emb / (np.linalg.norm(emb) + 1e-8)


# ═══════════════════════════════════════════════════════════
# AutoVC Model Components
# ═══════════════════════════════════════════════════════════

def _conv_bn(in_ch, out_ch, k=5):
    return nn.Sequential(nn.Conv1d(in_ch, out_ch, k, padding=k // 2),
                         nn.BatchNorm1d(out_ch), nn.ReLU())


class ContentEncoder(nn.Module):
    """Content encoder with narrow information bottleneck.

    Deliberately capacity-limited (dim_neck channels, one code every `freq` frames)
    so speaker identity can't leak through.
    """

    def __init__(self, dim_neck=16, dim_emb=192, dim_pre=256, freq=16):
        super().__init__()
        self.dim_neck = dim_neck
        self.freq = freq
        self.convs = nn.Sequential(_conv_bn(N_MELS + dim_emb, dim_pre),
                                   _conv_bn(dim_pre, dim_pre),
                                   _conv_bn(dim_pre, dim_pre))
        self.lstm = nn.LSTM(dim_pre, dim_neck, 2, batch_first=True, bidirectional=True)

    def forward(self, mel, speaker_emb):
        """
        Args:
            mel: (batch, 80, frames) normalized mel, frames divisible by freq
            speaker_emb: (batch, dim_emb) speaker embedding

        Returns:
            codes: (batch, frames // freq, dim_neck*2) content codes (bottleneck)
        """
        emb = speaker_emb.unsqueeze(-1).expand(-1, -1, mel.shape[-1])
        x = self.convs(torch.cat([mel, emb], dim=1)).transpose(1, 2)  # (batch, frames, dim_pre)

        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)  # (batch, frames, 2*dim_neck)
        fwd, bwd = out[..., :self.dim_neck], out[..., self.dim_neck:]

        # AutoVC downsampling: forward state at the END of each block, backward state
        # at the START, so each code summarizes its whole `freq`-frame block.
        return torch.cat([fwd[:, self.freq - 1::self.freq], bwd[:, ::self.freq]], dim=-1)


class Decoder(nn.Module):
    """Upsamples content codes to the frame rate and decodes them with the target
    speaker embedding into a mel-spectrogram."""

    def __init__(self, dim_neck=16, dim_emb=192, dim_pre=256, dim_lstm=512, freq=16):
        super().__init__()
        self.freq = freq
        self.lstm1 = nn.LSTM(dim_neck * 2 + dim_emb, dim_pre, 1, batch_first=True)
        self.convs = nn.Sequential(_conv_bn(dim_pre, dim_pre),
                                   _conv_bn(dim_pre, dim_pre),
                                   _conv_bn(dim_pre, dim_pre))
        self.lstm2 = nn.LSTM(dim_pre, dim_lstm, 2, batch_first=True)
        self.proj = nn.Linear(dim_lstm, N_MELS)

    def forward(self, codes, target_emb):
        """
        Args:
            codes: (batch, frames // freq, dim_neck*2)
            target_emb: (batch, dim_emb)

        Returns:
            mel_out: (batch, 80, frames)
        """
        up = codes.repeat_interleave(self.freq, dim=1)  # back to (batch, frames, 2*dim_neck)
        emb = target_emb.unsqueeze(1).expand(-1, up.shape[1], -1)
        x = torch.cat([up, emb], dim=-1)

        self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)
        x = self.convs(x.transpose(1, 2)).transpose(1, 2)
        self.lstm2.flatten_parameters()
        x, _ = self.lstm2(x)
        return self.proj(x).transpose(1, 2)


class Postnet(nn.Module):
    """5-layer conv postnet predicting a residual refinement of the decoder mel."""

    def __init__(self, dim=256, layers=5):
        super().__init__()
        chans = [N_MELS] + [dim] * (layers - 1) + [N_MELS]
        blocks = []
        for i in range(layers):
            blocks += [nn.Conv1d(chans[i], chans[i + 1], 5, padding=2), nn.BatchNorm1d(chans[i + 1])]
            if i < layers - 1:
                blocks.append(nn.Tanh())
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class AutoVC(nn.Module):
    """AutoVC: autoencoder for voice conversion with disentangled content/speaker.

    Works on normalized mels: (log_mel - mel_mean) / mel_std. The statistics are
    buffers so they are saved in the checkpoint with the weights.
    """

    def __init__(self, dim_neck=16, dim_emb=192, dim_pre=256, freq=16, dim_lstm=512):
        super().__init__()
        assert dim_emb == EMBEDDING_DIM, (
            f"AutoVC speaker embedding dim ({dim_emb}) must match "
            f"Module 2 embedding dim ({EMBEDDING_DIM}). "
            f"This is a cross-module integration requirement."
        )
        self.freq = freq
        self.encoder = ContentEncoder(dim_neck, dim_emb, dim_pre, freq)
        self.decoder = Decoder(dim_neck, dim_emb, dim_pre, dim_lstm, freq)
        self.postnet = Postnet(dim_pre)
        self.register_buffer('mel_mean', torch.zeros(N_MELS, 1))
        self.register_buffer('mel_std', torch.ones(N_MELS, 1))

    def normalize(self, log_mel):
        return (log_mel - self.mel_mean) / self.mel_std

    def denormalize(self, mel):
        return mel * self.mel_std + self.mel_mean

    def pad_to_freq(self, mel):
        """Right-pad normalized mel (with the silence value) to a multiple of freq."""
        pad = (-mel.shape[-1]) % self.freq
        if pad:
            silence = self.normalize(torch.full((N_MELS, 1), LOG_MEL_FLOOR, device=mel.device))
            mel = torch.cat([mel, silence.expand(mel.shape[0], -1, pad)], dim=-1)
        return mel

    def forward(self, mel, source_emb, target_emb):
        """
        Args:
            mel: (batch, 80, frames) NORMALIZED source mel
            source_emb: (batch, dim_emb) source speaker embedding
            target_emb: (batch, dim_emb) target speaker embedding

        Returns:
            mel_out: (batch, 80, frames) decoder output (normalized)
            mel_post: (batch, 80, frames) decoder output + postnet (normalized)
            codes: (batch, frames // freq, dim_neck*2) content codes
        """
        frames = mel.shape[-1]
        mel = self.pad_to_freq(mel)
        codes = self.encoder(mel, source_emb)
        mel_out = self.decoder(codes, target_emb)
        mel_post = mel_out + self.postnet(mel_out)
        return mel_out[..., :frames], mel_post[..., :frames], codes


# ═══════════════════════════════════════════════════════════
# Loss Functions (AutoVC paper, Eq. 6-8)
# ═══════════════════════════════════════════════════════════

def self_reconstruction_loss(mel_out, mel_post, mel_target):
    """Self-reconstruction loss (target speaker = source speaker): MSE of both the
    decoder output and the postnet output against the input mel."""
    return F.mse_loss(mel_post, mel_target) + F.mse_loss(mel_out, mel_target)


def content_consistency_loss(model, mel_post, codes, source_emb):
    """Content-code consistency loss: re-encoding the reconstruction must give the
    same content codes as the input (L1)."""
    codes_recon = model.encoder(model.pad_to_freq(mel_post), source_emb)
    return F.l1_loss(codes_recon, codes)


# ═══════════════════════════════════════════════════════════
# HiFi-GAN Vocoder Wrapper
# ═══════════════════════════════════════════════════════════

class HiFiGANVocoder:
    """Pretrained 16 kHz HiFi-GAN (speechbrain/tts-hifigan-libritts-16kHz).

    Downloaded once into checkpoints/hifigan-libritts-16kHz. If it cannot be loaded
    (e.g. offline on first run) we fall back to Griffin-Lim and say so loudly.
    """

    SOURCE = 'speechbrain/tts-hifigan-libritts-16kHz'

    def __init__(self, savedir=None):
        self.savedir = savedir or os.path.join(CHECKPOINT_DIR, 'hifigan-libritts-16kHz')
        self.model = None
        self.backend = None

    def load(self):
        try:
            from speechbrain.inference.vocoders import HIFIGAN
            from speechbrain.utils.fetching import LocalStrategy
            self.model = HIFIGAN.from_hparams(source=self.SOURCE, savedir=self.savedir,
                                              local_strategy=LocalStrategy.COPY)
            self.backend = 'hifigan'
            print(f"[HiFi-GAN] Loaded {self.SOURCE}")
        except Exception as e:  # noqa: BLE001
            self.backend = 'griffin-lim'
            print(f"[HiFi-GAN] WARNING: could not load vocoder ({e}); using Griffin-Lim fallback")

    def mel_to_audio(self, log_mel):
        """(80, frames) log magnitude mel (audio_to_mel format) -> waveform numpy array."""
        if self.backend is None:
            self.load()
        if isinstance(log_mel, np.ndarray):
            log_mel = torch.from_numpy(log_mel)
        log_mel = log_mel.float()

        if self.backend == 'hifigan':
            with torch.no_grad():
                return self.model.decode_batch(log_mel.unsqueeze(0)).squeeze().cpu().numpy()

        return librosa.feature.inverse.mel_to_audio(
            np.exp(log_mel.numpy()), sr=SAMPLE_RATE, n_fft=1024, hop_length=256,
            win_length=1024, power=1.0, fmin=0, fmax=8000)


# ═══════════════════════════════════════════════════════════
# Public API — Interface Contract (Section 5)
# ═══════════════════════════════════════════════════════════

_model = None
_vocoder = None


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'voice_conversion.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_model(config):
    m = config['model']
    return AutoVC(dim_neck=m.get('dim_neck', 16), dim_emb=m.get('dim_emb', EMBEDDING_DIM),
                  dim_pre=m.get('dim_pre', 256), freq=m.get('freq', 16),
                  dim_lstm=m.get('dim_lstm', 512))


def _load_model(checkpoint_path=None):
    """Load the AutoVC model with trained weights."""
    model = build_model(load_config())
    checkpoint_path = checkpoint_path or os.path.join(CHECKPOINT_DIR, 'voice_conversion.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No voice conversion checkpoint at {checkpoint_path}. Train it with "
            f"`python training/train_voice_conversion.py` first.")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state)
    print(f"[VoiceConversion] Loaded checkpoint from {checkpoint_path}")
    model.eval()
    return model


def _get_model():
    global _model
    if _model is None:
        _model = _load_model()
    return _model


def get_vocoder():
    global _vocoder
    if _vocoder is None:
        _vocoder = HiFiGANVocoder()
    return _vocoder


def backend():
    return load_config().get('backend', 'pretrained')


_openvoice = None


def _get_openvoice():
    global _openvoice
    if _openvoice is None:
        from third_party.openvoice.converter import ToneColorConverter
        _openvoice = ToneColorConverter(os.path.join(CHECKPOINT_DIR, 'pretrained', 'openvoice_v2'))
        print("[VoiceConversion] Loaded pretrained OpenVoice v2 tone-colour converter")
    return _openvoice


def _speech_only(audio):
    """Drop silent frames (energy VAD) so speaker embeddings describe the voice, not silence."""
    from models.diarization import speech_activity
    vad = speech_activity(audio, SAMPLE_RATE)
    hop = SAMPLE_RATE // 100
    mask = np.repeat(vad, hop)[:len(audio)]
    speech = audio[:len(mask)][mask]
    return speech if len(speech) > SAMPLE_RATE // 2 else audio


def _convert_openvoice(source_audio, target_speaker_ref):
    """Zero-shot conversion with OpenVoice v2 (works from a few seconds of reference)."""
    ov = _get_openvoice()
    tau = load_config().get('pretrained', {}).get('tau', 0.3)
    src_se = ov.speaker_embedding(_speech_only(source_audio), SAMPLE_RATE)
    tgt_se = ov.speaker_embedding(_speech_only(target_speaker_ref), SAMPLE_RATE)
    audio = ov.convert(source_audio, SAMPLE_RATE, src_se, tgt_se, tau=tau)
    audio = np.pad(audio, (0, max(0, len(source_audio) - len(audio))))[:len(source_audio)]
    return (audio / (np.abs(audio).max() + 1e-8) * 0.9).astype(np.float32)


def convert(source_audio: np.ndarray, target_speaker_ref: np.ndarray, sr: int) -> np.ndarray:
    """Converts source_audio into the voice of the speaker in target_speaker_ref.

    backend 'pretrained' (default): OpenVoice v2 tone-colour converter (zero-shot).
    backend 'custom': our AutoVC, which calls diarization.get_embedding() on
    target_speaker_ref (Module 2's embedding is its speaker encoder).

    Args:
        source_audio: Source speaker's audio waveform (numpy array).
        target_speaker_ref: Reference audio of the target speaker (numpy array).
        sr: Sample rate of both audio inputs.

    Returns:
        Converted waveform as numpy array at 16kHz, same duration as source_audio.
    """
    source_audio = np.asarray(source_audio, dtype=np.float32)
    target_speaker_ref = np.asarray(target_speaker_ref, dtype=np.float32)
    if sr != SAMPLE_RATE:
        source_audio = librosa.resample(source_audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        target_speaker_ref = librosa.resample(target_speaker_ref, orig_sr=sr, target_sr=SAMPLE_RATE)

    if backend() == 'pretrained':
        return _convert_openvoice(source_audio, target_speaker_ref)

    model = _get_model()

    # Speaker embeddings from Module 2's trained model (cross-module integration)
    src_emb = torch.from_numpy(speaker_embedding(source_audio)).unsqueeze(0)
    tgt_emb = torch.from_numpy(speaker_embedding(target_speaker_ref)).unsqueeze(0)

    # Peak-normalize the source so its mel level matches training data
    source_audio = source_audio / (np.abs(source_audio).max() + 1e-8) * 0.9
    mel = model.normalize(torch.from_numpy(audio_to_mel(source_audio)).unsqueeze(0))

    with torch.no_grad():
        _, mel_post, _ = model(mel, src_emb, tgt_emb)
    log_mel = model.denormalize(mel_post[0])

    audio = get_vocoder().mel_to_audio(log_mel)[:len(source_audio)]
    return (audio / (np.abs(audio).max() + 1e-8) * 0.9).astype(np.float32)


def modulate(audio: np.ndarray, sr: int, pitch_factor: float) -> np.ndarray:
    """DSP pitch-shift (librosa/pyrubberband), independent of the trained models.

    Args:
        audio: Audio waveform as numpy array.
        sr: Sample rate.
        pitch_factor: Pitch shift in semitones (e.g. +2.0 = up 2 semitones,
                      -3.0 = down 3 semitones).

    Returns:
        Pitch-shifted audio as numpy array.
    """
    if pitch_factor == 0:
        return np.asarray(audio)
    try:
        import pyrubberband as pyrb
        shifted = pyrb.pitch_shift(audio, sr, pitch_factor)
    except Exception:
        # Fallback to librosa if pyrubberband / rubberband CLI is not installed
        shifted = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=pitch_factor)
    return shifted
