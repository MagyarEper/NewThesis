#!/bin/bash
# =============================================================
# Run all Whisper fine-tune experiments on remote GPU server
# =============================================================

set -e

cd /home/makais/Thesis/NewThesis/NewThesis/

echo "=============================================="
echo "  Whisper Fine-tune Experiments"
echo "  Hungarian Dysarthric Speech Recognition"
echo "=============================================="

# ---- Install dependencies ----
echo ""
echo "[0/4] Installing dependencies..."
pip install --quiet transformers datasets evaluate peft accelerate jiwer soundfile
pip install --quiet torch torchaudio  # should already be there

# ---- Experiment 0: Baseline ----
echo ""
echo "=============================================="
echo "[1/4] Experiment 0: Baseline (no fine-tune)"
echo "=============================================="
python whisper_evaluate.py \
    --test-manifest test_manifest.csv \
    --model-name openai/whisper-small \
    --output-csv results/exp0_baseline.csv

# ---- Generate synthetic training data (if not already done) ----
if [ ! -f "synthetic_train_manifest.csv" ]; then
    echo ""
    echo "=============================================="
    echo "[GEN] Generating synthetic training data..."
    echo "=============================================="
    conda run -n grad-tts2 python generate_test_set.py \
        --checkpoint Grad-TTS/logs/hungarian_dysarthria/grad_500.pt \
        --manifest train_manifest.csv \
        --output-dir synthetic_train_wavs \
        --output-manifest synthetic_train_manifest.csv \
        --timesteps 10 \
        --temperature 1.2
fi

# ---- Experiment 1: Real data ----
echo ""
echo "=============================================="
echo "[2/4] Experiment 1: Fine-tune on REAL data"
echo "=============================================="
python whisper_finetune.py \
    --train-manifest train_manifest.csv \
    --val-manifest val_manifest.csv \
    --experiment real \
    --output-dir whisper_finetuned/exp1_real \
    --epochs 15 \
    --batch-size 8 \
    --lr 1e-4

python whisper_evaluate.py \
    --test-manifest test_manifest.csv \
    --model-path whisper_finetuned/exp1_real/best_model \
    --output-csv results/exp1_real.csv

# ---- Experiment 2: Synthetic data only ----
echo ""
echo "=============================================="
echo "[3/4] Experiment 2: Fine-tune on SYNTHETIC data"
echo "=============================================="
python whisper_finetune.py \
    --train-manifest synthetic_train_manifest.csv \
    --val-manifest val_manifest.csv \
    --experiment synthetic \
    --output-dir whisper_finetuned/exp2_synthetic \
    --epochs 15 \
    --batch-size 8 \
    --lr 1e-4

python whisper_evaluate.py \
    --test-manifest test_manifest.csv \
    --model-path whisper_finetuned/exp2_synthetic/best_model \
    --output-csv results/exp2_synthetic.csv

# ---- Experiment 3: Data Augmentation (Real + Synthetic) ----
echo ""
echo "=============================================="
echo "[4/4] Experiment 3: Fine-tune on REAL + SYNTHETIC"
echo "=============================================="
python whisper_finetune.py \
    --train-manifest train_manifest.csv \
    --synth-manifest synthetic_train_manifest.csv \
    --val-manifest val_manifest.csv \
    --experiment daug \
    --output-dir whisper_finetuned/exp3_daug \
    --epochs 15 \
    --batch-size 8 \
    --lr 1e-4

python whisper_evaluate.py \
    --test-manifest test_manifest.csv \
    --model-path whisper_finetuned/exp3_daug/best_model \
    --output-csv results/exp3_daug.csv

# ---- Summary ----
echo ""
echo "=============================================="
echo "  ALL EXPERIMENTS COMPLETE!"
echo "=============================================="
echo ""
echo "Results:"
for f in results/exp*_summary.txt; do
    echo "--- $(basename $f) ---"
    cat "$f"
    echo ""
done
