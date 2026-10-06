"""
Minimal wrapper around the OpenVoice v2 tone-colour converter (MIT license, see LICENSE).

Only the converter network is vendored (models.py, modules.py, attentions.py,
commons.py, transforms.py, mel_processing.py from github.com/myshell-ai/OpenVoice);
OpenVoice's TTS, text front-end, Whisper-based segmentation and watermarking are not
used, so none of its pinned dependencies are needed.

Weights: huggingface.co/myshell-ai/OpenVoiceV2 (converter/config.json, converter/checkpoint.pth)
"""

import json
import os

import librosa
import numpy as np
import torch

from .mel_processing import spectrogram_torch
from .models import SynthesizerTrn

HF_REPO = 'myshell-ai/OpenVoiceV2'


class ToneColorConverter:
    def __init__(self, ckpt_dir, device='cpu'):
        config_path = os.path.join(ckpt_dir, 'converter', 'config.json')
        ckpt_path = os.path.join(ckpt_dir, 'converter', 'checkpoint.pth')
        if not os.path.exists(ckpt_path):
            from huggingface_hub import hf_hub_download
            for f in ('converter/config.json', 'converter/checkpoint.pth'):
                hf_hub_download(HF_REPO, f, local_dir=ckpt_dir)

        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        self.data = cfg['data']
        self.sr = self.data['sampling_rate']  # 22050
        self.device = device
        self.model = SynthesizerTrn(0, self.data['filter_length'] // 2 + 1,
                                    n_speakers=self.data['n_speakers'], **cfg['model']).to(device).eval()
        state = torch.load(ckpt_path, map_location=device)['model']
        self.model.load_state_dict(state, strict=False)

    def _spec(self, audio):
        y = torch.from_numpy(np.asarray(audio, dtype=np.float32)).to(self.device)[None]
        d = self.data
        return spectrogram_torch(y, d['filter_length'], self.sr, d['hop_length'],
                                 d['win_length'], center=False)

    @torch.no_grad()
    def speaker_embedding(self, audio, sr, chunk_sec=10.0):
        """Tone-colour embedding (1, 256, 1); averaged over <=10 s chunks."""
        audio = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=sr, target_sr=self.sr)
        step = int(chunk_sec * self.sr)
        chunks = [audio[i:i + step] for i in range(0, len(audio), step)]
        chunks = [c for c in chunks if len(c) > self.sr // 2] or [audio]
        embs = [self.model.ref_enc(self._spec(c).transpose(1, 2)).unsqueeze(-1) for c in chunks]
        return torch.stack(embs).mean(0)

    @torch.no_grad()
    def convert(self, audio, sr, src_se, tgt_se, tau=0.3):
        """Convert `audio` (any sr) from src_se to tgt_se; returns audio at `sr`."""
        y = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=sr, target_sr=self.sr)
        spec = self._spec(y)
        lengths = torch.LongTensor([spec.size(-1)]).to(self.device)
        out = self.model.voice_conversion(spec, lengths, sid_src=src_se, sid_tgt=tgt_se, tau=tau)[0][0, 0]
        return librosa.resample(out.cpu().numpy(), orig_sr=self.sr, target_sr=sr)
