"""
Master training and evaluation script.

Runs all three modules sequentially:
  1. Train Conv-TasNet (Module 1 — Separation)
  2. Evaluate Module 1 
  3. Train ECAPA-TDNN (Module 2 — Diarization)
  4. Generate t-SNE visualization
  5. Evaluate Module 2
  6. Train AutoVC (Module 3 — Voice Conversion)
  7. Evaluate Module 3
  8. Run unit tests
"""

import os
import sys
import time
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)


if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def run_step(description, command):
    """Run a training/eval step and report timing."""
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"{'=' * 60}\n")
    
    start = time.time()
    result = subprocess.run(command, shell=True, cwd=PROJECT_ROOT)
    elapsed = time.time() - start
    
    status = "[PASS]" if result.returncode == 0 else "[FAIL]"
    print(f"\n{status} - {description} ({elapsed:.1f}s)")
    
    return result.returncode == 0


def main():
    print("=" * 60)
    print("  COCKTAIL PARTY PROBLEM - FULL TRAINING PIPELINE")
    print("=" * 60)
    
    total_start = time.time()
    results = {}
    
    py = f'"{sys.executable}"'

    # ─── Data (speaker split, test mixtures, diarization test set) ───
    if not os.path.exists(os.path.join('data', 'manifest.json')):
        results['data'] = run_step("Preparing data", f"{py} scripts/prepare_data.py")

    # ─── Module 1: Separation ───
    results['m1_train'] = run_step(
        "Module 1: Training Conv-TasNet Separation",
        f"{py} training/train_separation.py --config configs/separation.yaml"
    )
    
    results['m1_eval'] = run_step(
        "Module 1: Evaluating Separation (SI-SDRi)",
        f"{py} evaluation/eval_separation.py"
    )
    
    # ─── Module 2: Diarization ───
    results['m2_train'] = run_step(
        "Module 2: Training ECAPA-TDNN Diarization",
        f"{py} training/train_diarization.py --config configs/diarization.yaml"
    )
    
    results['m2_tsne'] = run_step(
        "Module 2: Generating t-SNE Visualization",
        f"{py} training/train_diarization.py --config configs/diarization.yaml --tsne"
    )
    
    results['m2_eval'] = run_step(
        "Module 2: Evaluating Diarization (DER)",
        f"{py} evaluation/eval_diarization.py"
    )
    
    # ─── Module 3: Voice Conversion ───
    results['m3_train'] = run_step(
        "Module 3: Training AutoVC Voice Conversion",
        f"{py} training/train_voice_conversion.py --config configs/voice_conversion.yaml"
    )
    
    results['m3_eval'] = run_step(
        "Module 3: Evaluating Voice Conversion (Speaker Similarity)",
        f"{py} evaluation/eval_voice_conversion.py"
    )
    
    # ─── Unit Tests ───
    results['tests'] = run_step(
        "Running Unit Tests",
        f"{py} -m pytest tests/ -v --tb=short"
    )
    
    # ─── Summary ───
    total_elapsed = time.time() - total_start
    
    print(f"\n{'=' * 60}")
    print(f"  PIPELINE SUMMARY ({total_elapsed:.0f}s total)")
    print(f"{'=' * 60}")
    
    for key, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {key}")
    
    # Check required deliverables
    print(f"\n{'-' * 60}")
    print("  Required Deliverables:")
    deliverables = [
        ("checkpoints/separation.pt", "Module 1 checkpoint"),
        ("checkpoints/diarization.pt", "Module 2 checkpoint"),
        ("checkpoints/voice_conversion.pt", "Module 3 checkpoint"),
        ("results/loss_curves/separation_loss.png", "Module 1 loss curve"),
        ("results/loss_curves/diarization_loss_acc.png", "Module 2 loss/acc curves"),
        ("results/loss_curves/voice_conversion_loss.png", "Module 3 loss curve"),
        ("results/metrics/separation_sisdri.csv", "SI-SDRi metrics"),
        ("results/metrics/tsne_embeddings.png", "t-SNE embeddings"),
        ("results/metrics/diarization_der.csv", "DER metrics"),
        ("results/metrics/speaker_verification.csv", "Speaker verification EER"),
        ("results/metrics/speaker_similarity.csv", "Speaker similarity metrics"),
    ]
    
    all_exist = True
    for path, desc in deliverables:
        exists = os.path.exists(path)
        status = "[PASS]" if exists else "[FAIL]"
        if not exists:
            all_exist = False
        print(f"  {status} {path} - {desc}")
    
    if all_exist:
        print(f"\n  [SUCCESS] All deliverables generated successfully!")
    else:
        print(f"\n  [WARNING] Some deliverables are missing.")
    
    print(f"\nTotal pipeline time: {total_elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
