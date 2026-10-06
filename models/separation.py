"""
Module 1 — Speech Separation (Conv-TasNet)

Separates a single-channel 2-speaker mixture into two per-speaker waveforms.
No dependency on Module 2 or Module 3.

Architecture: Conv-TasNet
  - Encoder: 1D conv (replaces STFT)
  - Separator: stack of dilated TCN blocks estimating per-speaker masks
  - Decoder: 1D transposed conv reconstructing each waveform

Reference: Luo & Mesgarani, "Conv-TasNet," arxiv.org/abs/1809.07454
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml

# ─── Canonical pipeline sample rate ───
SAMPLE_RATE = 16000


# ═══════════════════════════════════════════════════════════
# Model Components
# ═══════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """1D convolutional encoder — replaces STFT with a learned representation."""

    def __init__(self, kernel_size=16, num_feats=512):
        super().__init__()
        self.conv = nn.Conv1d(1, num_feats, kernel_size=kernel_size,
                              stride=kernel_size // 2, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, 1, time)
        return self.relu(self.conv(x))  # (batch, num_feats, frames)


class Decoder(nn.Module):
    """1D transposed convolutional decoder — reconstructs waveform from masked representation."""

    def __init__(self, kernel_size=16, num_feats=512):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(num_feats, 1,
                                         kernel_size=kernel_size,
                                         stride=kernel_size // 2, bias=False)

    def forward(self, x):
        # x: (batch, num_feats, frames)
        return self.deconv(x)  # (batch, 1, time)


class DepthwiseSeparableConv(nn.Module):
    """Conv-TasNet 1-D conv block: 1x1 (B->H) -> depthwise dilated conv (H) -> 1x1 back to
    B (residual) and to Sc (skip). Matches Fig. 1(c) of the paper."""

    def __init__(self, in_channels, hidden_channels, kernel_size, dilation, skip_channels):
        super().__init__()
        self.conv_in = nn.Conv1d(in_channels, hidden_channels, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = nn.GroupNorm(1, hidden_channels)  # gLN
        self.dconv = nn.Conv1d(hidden_channels, hidden_channels,
                               kernel_size=kernel_size, dilation=dilation,
                               padding=(kernel_size - 1) * dilation // 2,
                               groups=hidden_channels)
        self.prelu2 = nn.PReLU()
        self.norm2 = nn.GroupNorm(1, hidden_channels)
        self.res_conv = nn.Conv1d(hidden_channels, in_channels, 1)
        self.skip_conv = nn.Conv1d(hidden_channels, skip_channels, 1)

    def forward(self, x):
        out = self.norm1(self.prelu1(self.conv_in(x)))
        out = self.norm2(self.prelu2(self.dconv(out)))
        return x + self.res_conv(out), self.skip_conv(out)


class TCNSeparator(nn.Module):
    """Temporal Convolutional Network separator estimating per-speaker masks.

    num_feats (N): encoder channels, bottleneck (B): msk_num_feats,
    num_hidden (H): msk_num_hidden_feats.
    """

    def __init__(self, num_feats=512, bottleneck=128, num_hidden=512, num_layers=8,
                 num_stacks=3, num_sources=2, msk_kernel_size=3, mask_act='relu'):
        super().__init__()
        self.num_sources = num_sources
        self.input_norm = nn.GroupNorm(1, num_feats)
        self.input_conv = nn.Conv1d(num_feats, bottleneck, 1)

        self.tcn_blocks = nn.ModuleList([
            DepthwiseSeparableConv(bottleneck, num_hidden, msk_kernel_size,
                                   dilation=2 ** l, skip_channels=bottleneck)
            for _ in range(num_stacks) for l in range(num_layers)
        ])

        self.output_prelu = nn.PReLU()
        self.output_conv = nn.Conv1d(bottleneck, num_feats * num_sources, 1)
        self.mask_act = nn.ReLU() if mask_act == 'relu' else nn.Sigmoid()

    def forward(self, x):
        # x: (batch, num_feats, frames)
        batch, num_feats, frames = x.shape
        out = self.input_conv(self.input_norm(x))

        skip_sum = 0
        for block in self.tcn_blocks:
            out, skip = block(out)
            skip_sum = skip_sum + skip

        masks = self.mask_act(self.output_conv(self.output_prelu(skip_sum)))
        return masks.view(batch, self.num_sources, num_feats, frames)


class ConvTasNet(nn.Module):
    """Conv-TasNet: end-to-end time-domain speech separation."""

    def __init__(self, enc_kernel_size=16, enc_num_feats=512,
                 msk_kernel_size=3, msk_num_feats=128,
                 msk_num_hidden_feats=512, msk_num_layers=8,
                 msk_num_stacks=3, num_sources=2, msk_activate='relu'):
        super().__init__()
        self.enc_kernel_size = enc_kernel_size
        self.encoder = Encoder(kernel_size=enc_kernel_size, num_feats=enc_num_feats)
        self.separator = TCNSeparator(
            num_feats=enc_num_feats,
            bottleneck=msk_num_feats,
            num_hidden=msk_num_hidden_feats,
            num_layers=msk_num_layers,
            num_stacks=msk_num_stacks,
            num_sources=num_sources,
            msk_kernel_size=msk_kernel_size,
            mask_act=msk_activate,
        )
        self.decoder = Decoder(kernel_size=enc_kernel_size, num_feats=enc_num_feats)
        self.num_sources = num_sources

    def forward(self, mixture):
        """
        Args:
            mixture: (batch, 1, time) waveform tensor

        Returns:
            sources: (batch, num_sources, time) separated waveforms
        """
        # Pad so the encoder stride divides the signal; output is trimmed back below
        time = mixture.shape[-1]
        stride = self.enc_kernel_size // 2
        if time < self.enc_kernel_size:
            pad = self.enc_kernel_size - time
        else:
            pad = (stride - (time - self.enc_kernel_size) % stride) % stride
        mixture = F.pad(mixture, (0, pad))

        # Encode
        enc_out = self.encoder(mixture)  # (batch, num_feats, frames)

        # Estimate masks
        masks = self.separator(enc_out)  # (batch, num_sources, num_feats, frames)

        # Apply masks and decode
        sources = []
        for i in range(self.num_sources):
            masked = enc_out * masks[:, i]  # (batch, num_feats, frames)
            decoded = self.decoder(masked)  # (batch, 1, time)
            sources.append(decoded)

        sources = torch.cat(sources, dim=1)  # (batch, num_sources, time)
        return sources[..., :time]


# ═══════════════════════════════════════════════════════════
# Loss: SI-SDR with Permutation Invariant Training (uPIT)
# ═══════════════════════════════════════════════════════════

def si_sdr(estimate, reference):
    """Scale-Invariant Signal-to-Distortion Ratio.

    SI-SDR = 10 * log10( ||s_target||^2 / ||e_noise||^2 )
    """
    # Zero-mean
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    reference = reference - reference.mean(dim=-1, keepdim=True)

    # s_target = <est, ref> * ref / ||ref||^2
    dot = torch.sum(estimate * reference, dim=-1, keepdim=True)
    s_target = dot * reference / (torch.sum(reference ** 2, dim=-1, keepdim=True) + 1e-8)

    e_noise = estimate - s_target

    si_sdr_val = 10 * torch.log10(
        torch.sum(s_target ** 2, dim=-1) / (torch.sum(e_noise ** 2, dim=-1) + 1e-8)
    )
    return si_sdr_val


def pit_loss(estimates, references):
    """Utterance-level Permutation Invariant Training loss using negative SI-SDR.

    Computes loss for both output-to-target assignments, uses whichever is lower.

    Args:
        estimates: (batch, num_sources, time)
        references: (batch, num_sources, time)

    Returns:
        loss: scalar (negative SI-SDR, averaged over batch and sources)
    """
    # Permutation 1: est[0]->ref[0], est[1]->ref[1]
    sdr_p1 = si_sdr(estimates[:, 0], references[:, 0]) + \
             si_sdr(estimates[:, 1], references[:, 1])

    # Permutation 2: est[0]->ref[1], est[1]->ref[0]
    sdr_p2 = si_sdr(estimates[:, 0], references[:, 1]) + \
             si_sdr(estimates[:, 1], references[:, 0])

    # Pick the better permutation per sample; average over the two sources
    best_sdr = torch.max(sdr_p1, sdr_p2) / 2

    # Minimize negative SI-SDR
    return -best_sdr.mean()


# ═══════════════════════════════════════════════════════════
# Public API — Interface Contract (Section 5)
# ═══════════════════════════════════════════════════════════

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'separation.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_model(model_cfg):
    return ConvTasNet(
        enc_kernel_size=model_cfg.get('enc_kernel_size', 16),
        enc_num_feats=model_cfg.get('enc_num_feats', 512),
        msk_kernel_size=model_cfg.get('msk_kernel_size', 3),
        msk_num_feats=model_cfg.get('msk_num_feats', 128),
        msk_num_hidden_feats=model_cfg.get('msk_num_hidden_feats', 512),
        msk_num_layers=model_cfg.get('msk_num_layers', 8),
        msk_num_stacks=model_cfg.get('msk_num_stacks', 3),
        num_sources=model_cfg.get('num_sources', 2),
        msk_activate=model_cfg.get('msk_activate', 'relu'),
    )


def _load_model(checkpoint_path=None):
    """Load the separation model selected by `backend` in configs/separation.yaml.

    - pretrained (default): Asteroid Conv-TasNet trained on the full Libri2Mix
      sep_clean 16 kHz set (huggingface.co/JorisCos/ConvTasNet_Libri2Mix_sepclean_16k).
      Same architecture as ours, but trained on ~212 h instead of our 81 CPU-minutes.
    - custom: our ConvTasNet trained by training/train_separation.py.
    """
    config = load_config()
    if config.get('backend', 'pretrained') == 'pretrained':
        from asteroid.models import ConvTasNet as AsteroidConvTasNet
        source = config.get('pretrained', {}).get('source', 'JorisCos/ConvTasNet_Libri2Mix_sepclean_16k')
        # Trained at 16 kHz (its `sample_rate` attribute is a stale 8000 default in the
        # uploaded config; 16 kHz input scores 15 dB SI-SDRi on our held-out test set).
        model = AsteroidConvTasNet.from_pretrained(source)
        print(f"[Separation] Loaded pretrained {source}")
        return model.eval()

    model = build_model(config['model'])

    if checkpoint_path is None:
        checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'separation.pt')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No separation checkpoint at {checkpoint_path}. Train it with "
            f"`python training/train_separation.py` first.")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state)
    print(f"[Separation] Loaded checkpoint from {checkpoint_path}")

    model.eval()
    return model


# Module-level cached model
_model = None


def separate(mixture_path: str) -> tuple:
    """Loads a mixture audio file, returns (stream1, stream2) waveforms
    at the model's native sample rate (16kHz).

    Args:
        mixture_path: Path to a single-channel mixture audio file.

    Returns:
        Tuple of (stream1, stream2) as numpy arrays at 16kHz.
    """
    import librosa as _librosa
    audio_np, _ = _librosa.load(mixture_path, sr=SAMPLE_RATE)  # mono, 16 kHz
    return separate_waveform(audio_np)


def separate_waveform(audio_np: np.ndarray) -> tuple:
    """Same as separate(), for a mono 16 kHz waveform already in memory."""
    global _model
    if _model is None:
        _model = _load_model()

    audio_np = np.asarray(audio_np, dtype=np.float32)
    waveform = torch.from_numpy(audio_np).view(1, 1, -1)  # (batch, 1, time)

    with torch.no_grad():
        sources = _model(waveform)  # (1, 2, time)

    # SI-SDR training leaves the output scale arbitrary: rescale each stream to the
    # mixture's peak level so it is audible and never clips when written to WAV.
    peak = float(np.abs(audio_np).max()) or 1.0
    streams = []
    for i in range(2):
        s = sources[0, i].cpu().numpy()
        s = s - s.mean()
        streams.append(s * (peak / (np.abs(s).max() + 1e-8)))

    return (streams[0], streams[1])
