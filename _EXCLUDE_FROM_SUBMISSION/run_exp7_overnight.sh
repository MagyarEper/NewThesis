#!/bin/bash
BASE=/home/makais/Thesis/NewThesis/NewThesis
cd $BASE

echo "============================================================"
echo "  Exp 7 — Szintetikus V2 only fine-tune"
echo "  Start: $(date)"
echo "============================================================"

python3 whisper_finetune.py \
    --train-manifest v2_aug_train_manifest.csv \
    --val-manifest val_manifest.csv \
    --experiment synthetic \
    --output-dir whisper_finetuned/exp7_synth_v2only \
    --epochs 15 \
    --batch-size 8 \
    --lr 1e-4

echo ""
echo "Fine-tune kész: $(date)"
echo ""
echo "Kiértékelés..."

python3 whisper_evaluate.py \
    --test-manifest test_manifest.csv \
    --model-path whisper_finetuned/exp7_synth_v2only/best_model \
    --output-csv results/exp7_synth_v2only.csv

echo ""
echo "============================================================"
echo "  KÉSZ! $(date)"
echo "  Model: whisper_finetuned/exp7_synth_v2only/best_model"
echo "  Test set: test_manifest.csv (1016 utt)"
echo "============================================================"
