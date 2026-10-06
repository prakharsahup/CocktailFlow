# Checkpoints directory
# This directory is gitignored — trained model .pt files go here.
# Expected checkpoints:
#   - separation.pt          (training/train_separation.py)
#   - diarization.pt         (training/train_diarization.py)
#   - voice_conversion.pt    (training/train_voice_conversion.py, needs diarization.pt)
#   - hifigan-libritts-16kHz/ (pretrained SpeechBrain HiFi-GAN, downloaded automatically
#                              on first use of models/voice_conversion.py)
