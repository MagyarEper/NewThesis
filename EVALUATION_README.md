# TTS Quality Evaluation - Quick Start

This directory contains scripts for evaluating the trained Hungarian dysarthria TTS model with intrusive metrics.

## 📋 Workflow

### 1. Install Dependencies
```bash
bash install_evaluation_deps.sh
```

This installs:
- **librosa** - audio processing
- **fastdtw** - DTW alignment for MCD
- **praat-parselmouth** - F0 extraction
- **pystoi** - P-STOI/ESTOI intelligibility metrics
- **pandas, numpy, scipy** - data processing

### 2. Generate Test Set Predictions

Generate synthetic audio for all 1,016 test utterances:

```bash
python generate_test_set.py \
    --checkpoint logs/hungarian_dysarthria/grad_500.pt \
    --manifest test_manifest.txt \
    --output-dir generated_test_wavs \
    --length-scale 1.0 \
    --temperature 1.2 \
    --timesteps 20
```

**Parameters (optimized from sanity check):**
- `length_scale=1.0` - Natural speaking rate
- `temperature=1.2` - Slightly increased diversity
- `timesteps=20` - Good quality/speed tradeoff (RTF ~0.10)

**Estimated time:** 1-2 hours for full test set

**Output:** `generated_test_wavs/` directory with 1,016 WAV files

### 3. Run Evaluation Metrics

Compute intrusive metrics on real vs. synthetic pairs:

```bash
python evaluate_metrics.py \
    --real-dir wavs_16khz \
    --synth-dir generated_test_wavs \
    --manifest test_manifest.txt \
    --output evaluation_results.csv
```

**Metrics computed:**
- **MCD** (Mel-Cepstral Distortion with DTW) - spectral similarity [dB, lower better]
- **F0 RMSE** - pitch accuracy on voiced frames [log Hz, lower better]
- **VUV Error** - voiced/unvoiced decision accuracy [%, lower better]
- **P-STOI** - short-time intelligibility [0-1, higher better]
- **ESTOI** - extended intelligibility [0-1, higher better]
- **PPG-D** - phonetic posteriorgram distance (placeholder, needs ASR model)

**Outputs:**
- `evaluation_results.csv` - per-utterance results
- `evaluation_results_summary.csv` - overall mean±std
- `evaluation_results_per_speaker.csv` - per-speaker breakdown

### 4. Analyze Results

Review the CSV files:

```bash
# View overall statistics
column -s, -t < evaluation_results_summary.csv

# View per-speaker breakdown (first 20 speakers)
head -n 21 evaluation_results_per_speaker.csv | column -s, -t

# Sort by MCD to find best/worst utterances
sort -t, -k2 -n evaluation_results.csv | head -n 10  # Best
sort -t, -k2 -rn evaluation_results.csv | head -n 10  # Worst
```

## 📊 Interpreting Results

### Expected Ranges (based on TTS literature)

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| MCD | < 5.0 dB | 5-7 dB | > 7 dB |
| F0 RMSE | < 0.2 log Hz | 0.2-0.4 | > 0.4 |
| VUV Error | < 5% | 5-10% | > 10% |
| STOI | > 0.85 | 0.75-0.85 | < 0.75 |
| ESTOI | > 0.80 | 0.70-0.80 | < 0.70 |

### Interpretation

- **MCD**: Measures spectral similarity. Lower = more natural-sounding.
- **F0 RMSE**: Pitch accuracy. Important for prosody and naturalness.
- **VUV Error**: How well the model distinguishes voiced/unvoiced segments.
- **STOI/ESTOI**: Intelligibility metrics. Higher = easier to understand.

### Analysis Tips

1. **Overall quality**: Check mean±std in summary file
2. **Speaker consistency**: Compare per-speaker variance (high variance = inconsistent)
3. **Outliers**: Identify worst utterances for error analysis
4. **Correlations**: Check if MCD correlates with STOI (should be inverse)

## 🔧 Troubleshooting

### Issue: Missing dependencies
```bash
# Re-run installation with verbose output
pip install -r evaluation_requirements.txt -v
```

### Issue: CUDA out of memory during generation
```bash
# Generate in smaller batches (modify generate_test_set.py to add batch processing)
# Or use CPU (slower but works)
export CUDA_VISIBLE_DEVICES=""
python generate_test_set.py ...
```

### Issue: F0 extraction fails
```bash
# Check Praat-Parselmouth installation
python -c "import parselmouth; print(parselmouth.__version__)"
```

### Issue: STOI computation fails
```bash
# Check pystoi installation
python -c "from pystoi import stoi; print('OK')"
```

## 📝 Next Steps

After evaluation:

1. **Update DOKUMENTACIO.md** - Add Part 2 (Evaluation Results)
2. **Create presentation tables** - Format results for Monday presentation
3. **Identify improvement areas** - Which speakers/phonemes need work?
4. **Plan Hungarian phoneme dict** - Post-Monday improvement (expected 70%→85-90%)

## 📚 References

- **MCD**: [Kubichek 1993] "Mel-cepstral distance measure for objective speech quality assessment"
- **STOI**: [Taal et al. 2011] "An algorithm for intelligibility prediction of time-frequency weighted noisy speech"
- **PPG-D**: [Huybrechts et al. 2021] "Robust phonetic posteriorgram distance for multi-speaker TTS evaluation"
