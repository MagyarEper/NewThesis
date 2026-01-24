#!/bin/bash
# Install evaluation dependencies

echo "Installing evaluation metrics dependencies..."
echo "=============================================="

pip install -r evaluation_requirements.txt

echo ""
echo "✓ Installation complete"
echo ""
echo "Installed packages:"
echo "  - librosa (audio processing)"
echo "  - fastdtw (DTW for MCD)"
echo "  - praat-parselmouth (F0 extraction)"
echo "  - pystoi (intelligibility metrics)"
echo "  - pandas, numpy, scipy (data processing)"
echo ""
echo "You can now run:"
echo "  1. python generate_test_set.py --checkpoint logs/hungarian_dysarthria/grad_500.pt --manifest test_manifest.txt"
echo "  2. python evaluate_metrics.py --real-dir wavs_16khz --synth-dir generated_test_wavs --manifest test_manifest.txt"
