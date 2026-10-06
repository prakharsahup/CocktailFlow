"""
Tests for Module 2 — Speaker Embedding & Diarization

Verifies:
  - ECAPA-TDNN model instantiation and forward pass
  - Embedding dimensionality (must be 192 to match Module 3)
  - AAM-Softmax loss computation and prediction
  - DER computation (with optimal speaker mapping)
  - get_embedding() and diarize() API contracts (need a trained checkpoint)
"""

import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
from models.diarization import (
    ECAPATDNN, AAMSoftmax, extract_log_mel,
    get_embedding, diarize, SAMPLE_RATE, EMBEDDING_DIM
)
from evaluation.eval_diarization import compute_der

from models.diarization import backend

# pretrained backend downloads its weights; custom needs a trained checkpoint
needs_checkpoint = pytest.mark.skipif(
    backend() == 'custom' and not os.path.exists(os.path.join(ROOT, 'checkpoints', 'diarization.pt')),
    reason="no trained diarization checkpoint")


class TestECAPATDNN:
    """Test the ECAPA-TDNN architecture."""

    def test_model_instantiation(self):
        model = ECAPATDNN()
        assert model.embedding_dim == 192

    def test_forward_pass_shape(self):
        model = ECAPATDNN(n_mels=80, channels=256)
        x = torch.randn(2, 80, 200)  # (batch, n_mels, frames)
        assert model(x).shape == (2, EMBEDDING_DIM)

    def test_has_no_untrained_classifier_head(self):
        """Accuracy must come from the AAM-Softmax weights, not a separate untrained head."""
        assert not hasattr(ECAPATDNN(), 'classifier')

    def test_embedding_dim_matches_contract(self):
        """Embedding dim must be exactly EMBEDDING_DIM for Module 3 integration."""
        assert ECAPATDNN().embedding_dim == EMBEDDING_DIM


class TestAAMSoftmax:
    """Test the AAM-Softmax loss."""

    def test_loss_computation(self):
        loss_fn = AAMSoftmax(embedding_dim=192, num_classes=30)
        loss = loss_fn(torch.randn(4, 192), torch.LongTensor([0, 1, 2, 3]))
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_loss_lower_for_aligned_embeddings(self):
        """Loss should be lower when embeddings align with their class weights."""
        torch.manual_seed(0)
        loss_fn = AAMSoftmax(embedding_dim=192, num_classes=10)
        labels = torch.LongTensor([0, 1])
        aligned = loss_fn.weight[labels].detach().clone()
        assert loss_fn(aligned, labels).item() < loss_fn(torch.randn(2, 192), labels).item()

    def test_cosine_prediction(self):
        """argmax of AAMSoftmax.cosine is the class whose weight the embedding matches."""
        loss_fn = AAMSoftmax(embedding_dim=192, num_classes=10)
        emb = loss_fn.weight[[3, 7]].detach() * 5.0
        assert loss_fn.cosine(emb).argmax(1).tolist() == [3, 7]


class TestLogMel:
    """Test feature extraction."""

    def test_log_mel_shape(self):
        audio = np.random.randn(16000).astype(np.float32)
        mel = extract_log_mel(audio, 16000, n_mels=80)
        assert mel.shape[0] == 80
        assert mel.shape[1] > 0


class TestDER:
    def test_perfect_prediction(self):
        gt = [('A', 0.0, 2.0), ('B', 2.5, 4.0)]
        assert compute_der(gt, gt, 4.0)['der'] == pytest.approx(0.0)

    def test_label_names_do_not_matter(self):
        """Cluster ids are arbitrary: the optimal mapping makes renamed labels score 0."""
        gt = [('A', 0.0, 2.0), ('B', 2.5, 4.0)]
        pred = [('Speaker_1', 0.0, 2.0), ('Speaker_0', 2.5, 4.0)]
        assert compute_der(pred, gt, 4.0)['der'] == pytest.approx(0.0)

    def test_single_cluster_confuses_one_speaker(self):
        gt = [('A', 0.0, 2.0), ('B', 2.0, 4.0)]
        res = compute_der([('X', 0.0, 4.0)], gt, 4.0)
        assert res['confusion'] == pytest.approx(0.5)
        assert res['miss'] == pytest.approx(0.0)


@needs_checkpoint
class TestGetEmbedding:
    """Test the get_embedding() public API."""

    def test_returns_correct_shape(self):
        audio = np.random.randn(SAMPLE_RATE * 2).astype(np.float32)
        emb = get_embedding(audio, SAMPLE_RATE)
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (EMBEDDING_DIM,)

    def test_deterministic(self):
        audio = np.random.randn(SAMPLE_RATE * 2).astype(np.float32)
        np.testing.assert_array_almost_equal(get_embedding(audio, SAMPLE_RATE),
                                             get_embedding(audio, SAMPLE_RATE))


@needs_checkpoint
class TestDiarize:
    """Test the diarize() public API."""

    def test_returns_list_of_tuples(self):
        audio = np.random.randn(SAMPLE_RATE * 5).astype(np.float32)
        result = diarize(audio, SAMPLE_RATE)
        assert isinstance(result, list)
        for spk, start, end in result:
            assert isinstance(spk, str)
            assert isinstance(start, float) and isinstance(end, float)
            assert end > start

    def test_segments_do_not_overlap(self):
        conv = os.path.join(ROOT, 'data', 'diarization_test', 'conv_0000.wav')
        if not os.path.exists(conv):
            pytest.skip("run scripts/prepare_data.py")
        import soundfile as sf
        audio, sr = sf.read(conv, dtype='float32')
        result = diarize(audio, sr)
        assert len(result) >= 2
        for (_, _, end), (_, start, _) in zip(result, result[1:]):
            assert start >= end


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
