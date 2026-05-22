#!/bin/bash
# Run this on remote server (deep07)

cd /home/makais/Thesis/NewThesis/NewThesis/

# Pull latest changes
git pull origin main

# USE grad-tts2 conda environment (it has working PyTorch/torchaudio)
conda run -n grad-tts2 python generate_test_set.py \
    --checkpoint Grad-TTS/logs/hungarian_dysarthria/grad_500.pt \
    --manifest test_manifest.csv \
    --output-dir generated_test_wavs_FIXED \
    --timesteps 10

echo ""
echo "=========================================="
echo "Generation complete!"
echo "Now run evaluation:"
echo ""
echo "conda run -n grad-tts2 python evaluate_metrics.py \\"
echo "    --real-dir wavs_16khz \\"
echo "    --synth-dir generated_test_wavs_FIXED \\"
echo "    --manifest test_manifest.csv \\"
echo "    --output evaluation_FIXED_MEL_CLAMP.csv"
echo "=========================================="
