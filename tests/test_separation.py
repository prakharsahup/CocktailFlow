"""
Tests for Module 1 — Speech Separation

Verifies:
  - Model instantiation and forward pass
  - SI-SDR computation
  - PIT loss computation
  - separate() API contract
"""

import os
import sys
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.separation import ConvTasNet, si_sdr, pit_loss, separate, SAMPLE_RATE


class TestConvTasNet:
    """Test the Conv-TasNet model architecture."""

    def test_model_instantiation(self):
        model = ConvTasNet()
        assert model is not None
        assert model.num_sources == 2

    def test_forward_pass_shape(self):
        model = ConvTasNet()
        batch_size = 2
        time_steps = 16000  # 1 second at 16kHz
        x = torch.randn(batch_size, 1, time_steps)
        y = model(x)
        assert y.shape[0] == batch_size
        assert y.shape[1] == 2  # num_sources

    def test_output_length_matches_input(self):
        model = ConvTasNet()
        x = torch.randn(1, 1, 16000)
        y = model(x)
        assert y.shape[-1] == x.shape[-1]

    def test_output_length_odd_input(self):
        """Lengths not divisible by the encoder stride must still round-trip exactly."""
        model = ConvTasNet(enc_kernel_size=32, enc_num_feats=64, msk_num_feats=32,
                           msk_num_hidden_feats=64, msk_num_layers=2, msk_num_stacks=1)
        for n in (12345, 16001, 20):
            assert model(torch.randn(1, 1, n)).shape == (1, 2, n)


class TestSISDR:
    """Test SI-SDR computation."""

    def test_perfect_signal(self):
        """SI-SDR of a signal with itself should be very high."""
        signal = torch.randn(1, 16000)
        sdr = si_sdr(signal, signal)
        assert sdr.item() > 30  # should be very high (ideally inf)

    def test_orthogonal_signals(self):
        """SI-SDR of orthogonal signals should be very low."""
        s1 = torch.FloatTensor([[1.0, 1.0, -1.0, -1.0]])
        s2 = torch.FloatTensor([[1.0, -1.0, 1.0, -1.0]])
        sdr = si_sdr(s1, s2)
        assert sdr.item() < 0

    def test_scaled_signal(self):
        """SI-SDR should be scale-invariant."""
        signal = torch.randn(1, 16000)
        scaled = signal * 2.0
        sdr = si_sdr(scaled, signal)
        assert sdr.item() > 30


class TestPITLoss:
    """Test Permutation Invariant Training loss."""

    def test_pit_correct_permutation(self):
        """PIT should find the correct permutation."""
        s1 = torch.randn(1, 1, 8000)
        s2 = torch.randn(1, 1, 8000)
        estimates = torch.cat([s1, s2], dim=1)  # correct order
        references = torch.cat([s1, s2], dim=1)
        loss = pit_loss(estimates, references)
        assert loss.item() < -20  # should be very good

    def test_pit_swapped_permutation(self):
        """PIT should handle swapped sources."""
        s1 = torch.randn(1, 1, 8000)
        s2 = torch.randn(1, 1, 8000)
        estimates = torch.cat([s2, s1], dim=1)  # swapped
        references = torch.cat([s1, s2], dim=1)
        loss = pit_loss(estimates, references)
        assert loss.item() < -20  # should still be good


from models.separation import load_config

# pretrained backend downloads its weights; custom needs a trained checkpoint
HAS_CHECKPOINT = load_config().get('backend', 'pretrained') == 'pretrained' or os.path.exists(
    os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'separation.pt'))


@pytest.mark.skipif(not HAS_CHECKPOINT, reason="no trained separation checkpoint")
class TestSeparateAPI:
    """Test the separate() public API."""

    def test_separate_returns_two_arrays(self, tmp_path):
        """separate() returns two arrays as long as the input, within [-1, 1]."""
        import soundfile as sf
        test_audio = np.random.randn(SAMPLE_RATE * 2 + 7).astype(np.float32) * 0.1
        test_path = str(tmp_path / 'mixture.wav')
        sf.write(test_path, test_audio, SAMPLE_RATE)

        s1, s2 = separate(test_path)
        for s in (s1, s2):
            assert isinstance(s, np.ndarray)
            assert len(s) == len(test_audio)
            assert np.abs(s).max() <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
