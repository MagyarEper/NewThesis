#!/bin/bash
# Complete evaluation pipeline
# Usage: bash run_full_evaluation.sh

set -e  # Exit on error

echo "=============================================="
echo "TTS QUALITY EVALUATION - COMPLETE PIPELINE"
echo "=============================================="
echo ""

# Step 1: Check dependencies
echo "Step 1/3: Checking dependencies..."
echo "-----------------------------------"
if ! python -c "import librosa, fastdtw, parselmouth, pystoi" 2>/dev/null; then
    echo "Missing dependencies. Installing..."
    bash install_evaluation_deps.sh
else
    echo "✓ All dependencies installed"
fi
echo ""

# Step 2: Generate test set
echo "Step 2/3: Generating test set predictions..."
echo "----------------------------------------------"

CHECKPOINT="logs/hungarian_dysarthria/grad_500.pt"
MANIFEST="test_manifest.txt"
OUTPUT_DIR="generated_test_wavs"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Please ensure the checkpoint exists or update the path."
    exit 1
fi

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: Manifest not found: $MANIFEST"
    echo "Please ensure the test manifest exists."
    exit 1
fi

# Check if already generated
if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR | wc -l)" -gt 1000 ]; then
    echo "✓ Test set already generated ($(ls -A $OUTPUT_DIR | wc -l) files found)"
    read -p "Regenerate? (y/N): " confirm
    if [[ ! $confirm =~ ^[Yy]$ ]]; then
        echo "Skipping generation..."
    else
        echo "Generating..."
        python generate_test_set.py \
            --checkpoint "$CHECKPOINT" \
            --manifest "$MANIFEST" \
            --output-dir "$OUTPUT_DIR" \
            --length-scale 1.0 \
            --temperature 1.2 \
            --timesteps 20
    fi
else
    echo "Generating test set (this may take 1-2 hours)..."
    python generate_test_set.py \
        --checkpoint "$CHECKPOINT" \
        --manifest "$MANIFEST" \
        --output-dir "$OUTPUT_DIR" \
        --length-scale 1.0 \
        --temperature 1.2 \
        --timesteps 20
fi
echo ""

# Step 3: Run evaluation
echo "Step 3/3: Computing evaluation metrics..."
echo "------------------------------------------"

REAL_DIR="wavs_16khz"
SYNTH_DIR="$OUTPUT_DIR"
OUTPUT_CSV="evaluation_results.csv"

if [ ! -d "$REAL_DIR" ]; then
    echo "ERROR: Real audio directory not found: $REAL_DIR"
    exit 1
fi

python evaluate_metrics.py \
    --real-dir "$REAL_DIR" \
    --synth-dir "$SYNTH_DIR" \
    --manifest "$MANIFEST" \
    --output "$OUTPUT_CSV"

echo ""
echo "=============================================="
echo "EVALUATION COMPLETE"
echo "=============================================="
echo ""
echo "Results saved:"
echo "  - $OUTPUT_CSV (per-utterance)"
echo "  - evaluation_results_summary.csv (overall mean±std)"
echo "  - evaluation_results_per_speaker.csv (per-speaker breakdown)"
echo ""
echo "Quick view of results:"
echo "----------------------"
echo ""
cat evaluation_results_summary.csv | column -s, -t
echo ""
echo "For full analysis, see: EVALUATION_README.md"
echo "=============================================="
