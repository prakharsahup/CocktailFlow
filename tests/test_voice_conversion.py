"""
Tests for Module 3 — Voice Conversion & Modulation

Verifies:
  - AutoVC model instantiation and forward pass
  - Output frames == input frames (regression: codes must be upsampled)
  - Content encoder bottleneck
  - convert() calls diarization.get_embedding() (not a duplicate encoder)
  - modulate() works independently of trained models
  - Speaker embedding dimension assertion
"""

import os
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
from models.voice_conversion import (
    AutoVC, ContentEncoder, audio_to_mel, backend, convert, modulate,
    SAMPLE_RATE, EMBEDDING_DIM
)


def small_autovc():
    return AutoVC(dim_neck=16, dim_emb=EMBEDDING_DIM, dim_pre=64, freq=16, dim_lstm=64)


class TestAutoVC:
    """Test the AutoVC architecture."""

    def test_model_instantiation(self):
        assert AutoVC(dim_emb=EMBEDDING_DIM) is not None

    def test_forward_pass_shape(self):
        model = small_autovc()
        mel = torch.randn(1, 80, 256)  # (batch, n_mels, frames)
        emb = torch.randn(1, EMBEDDING_DIM)
        mel_out, mel_post, codes = model(mel, emb, emb)
        assert mel_out.shape == (1, 80, 256)
        assert mel_post.shape == (1, 80, 256)
        assert codes.shape == (1, 256 // 16, 32)

    @pytest.mark.parametrize('frames', [100, 128, 203])
    def test_output_frames_equal_input_frames(self, frames):
        """Regression: the decoder must upsample codes back to the input frame rate
        (previously the output was 1/freq of the input length)."""
        model = small_autovc()
        emb = torch.randn(2, EMBEDDING_DIM)
        mel_out, mel_post, _ = model(torch.randn(2, 80, frames), emb, emb)
        assert mel_out.shape[-1] == frames
        assert mel_post.shape[-1] == frames

    def test_embedding_dim_assertion(self):
        """Model should reject mismatched embedding dims."""
        with pytest.raises(AssertionError):
            AutoVC(dim_emb=256)  # Wrong dim — should be EMBEDDING_DIM (192)


class TestContentEncoder:
    """Test the content encoder bottleneck."""

    def test_bottleneck_shape(self):
        encoder = ContentEncoder(dim_neck=32, dim_emb=EMBEDDING_DIM, dim_pre=64, freq=32)
        codes = encoder(torch.randn(1, 80, 256), torch.randn(1, EMBEDDING_DIM))
        assert codes.shape == (1, 256 // 32, 32 * 2)  # one code per `freq` frames

    def test_bottleneck_is_narrow(self):
        """Bottleneck should be much smaller than input dim to force disentanglement."""
        assert ContentEncoder(dim_neck=16, dim_emb=EMBEDDING_DIM).dim_neck < 80


class TestMelSpectrogram:
    def test_audio_to_mel_shape(self):
        audio = np.random.randn(SAMPLE_RATE * 2).astype(np.float32)
        mel = audio_to_mel(audio, SAMPLE_RATE)
        assert mel.shape[0] == 80
        assert mel.shape[1] == SAMPLE_RATE * 2 // 256 + 1


class TestModulate:
    """Test DSP pitch modulation (independent of trained models)."""

    def test_modulate_returns_array(self):
        audio = np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.1
        shifted = modulate(audio, SAMPLE_RATE, pitch_factor=2.0)
        assert isinstance(shifted, np.ndarray)
        assert len(shifted) > 0

    def test_modulate_zero_shift(self):
        audio = np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.1
        np.testing.assert_array_equal(modulate(audio, SAMPLE_RATE, pitch_factor=0.0), audio)

    def test_modulate_positive_and_negative(self):
        audio = np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.1
        assert abs(len(modulate(audio, SAMPLE_RATE, 3.0)) - len(audio)) < 100
        assert abs(len(modulate(audio, SAMPLE_RATE, -3.0)) - len(audio)) < 100


class TestCrossModuleIntegration:
    """Test that Module 3 correctly uses Module 2's embedding model."""

    def test_convert_calls_get_embedding(self):
        """convert() should call diarization.get_embedding(), not a duplicate encoder."""
        with patch('models.voice_conversion.backend', return_value='custom'), \
                patch('models.voice_conversion.get_embedding') as mock_emb, \
                patch('models.voice_conversion._get_model', return_value=small_autovc().eval()), \
                patch('models.voice_conversion.get_vocoder') as mock_voc:
            mock_emb.return_value = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            mock_voc.return_value.mel_to_audio.side_effect = \
                lambda mel: np.zeros(mel.shape[-1] * 256, dtype=np.float32)

            source = np.random.randn(SAMPLE_RATE).astype(np.float32)
            target = np.random.randn(SAMPLE_RATE).astype(np.float32)
            convert(source, target, SAMPLE_RATE)

            # get_embedding called for both source and target
            assert mock_emb.call_count == 2


@pytest.mark.skipif(backend() == 'custom' and
                    not os.path.exists(os.path.join(ROOT, 'checkpoints', 'voice_conversion.pt')),
                    reason="no trained voice conversion checkpoint")
class TestConvert:
    def test_converted_length_matches_source(self):
        source = (np.random.randn(SAMPLE_RATE * 2 + 123) * 0.1).astype(np.float32)
        target = (np.random.randn(SAMPLE_RATE * 2) * 0.1).astype(np.float32)
        out = convert(source, target, SAMPLE_RATE)
        assert abs(len(out) - len(source)) <= 256  # at most one hop
        assert np.abs(out).max() <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
