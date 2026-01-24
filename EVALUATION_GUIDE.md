# 🎯 Evaluation Step-by-Step Guide (deep07)

## Prerequisites
- ✅ Git pull completed
- ✅ Training checkpoint exists: `logs/hungarian_dysarthria/grad_500.pt`
- ✅ Test manifest exists: `test_manifest.txt`
- ✅ Real audio exists: `wavs_16khz/`

---

## Step 1: Activate Environment

```bash
cd /home/makais/Thesis/NewThesis/NewThesis
source thesis/bin/activate
```

**Verify:**
```bash
which python
# Should show: /home/makais/Thesis/NewThesis/NewThesis/thesis/bin/python
```

---

## Step 2: Install Evaluation Dependencies

```bash
bash install_evaluation_deps.sh
```

**This installs:**
- librosa (audio processing)
- fastdtw (DTW for MCD)
- praat-parselmouth (F0 extraction)
- pystoi (intelligibility metrics)
- pandas, numpy, scipy

**Expected time:** 2-3 minutes

**Verify installation:**
```bash
python -c "import librosa, fastdtw, parselmouth, pystoi; print('✓ All dependencies OK')"
```

---

## Step 3: Check Files Exist

```bash
# Check checkpoint
ls -lh Grad-TTS/logs/hungarian_dysarthria/grad_500.pt

# Check test manifest
wc -l test_manifest.txt
# Should show: 1016 test_manifest.txt

# Check real audio directory
ls wavs_16khz/ | wc -l
# Should show: ~10000+ files
```

---

## Step 4: Generate Test Set (1-2 hours)

### Option A: Run in tmux (RECOMMENDED - can disconnect safely)

```bash
# Start tmux session
tmux new -s evaluation

# Run generation
python generate_test_set.py \
    --checkpoint Grad-TTS/logs/hungarian_dysarthria/grad_500.pt \
    --manifest test_manifest.txt \
    --output-dir generated_test_wavs \
    --length-scale 1.0 \
    --temperature 1.2 \
    --timesteps 20

# Detach from tmux: Press Ctrl+B, then D
# Reattach later: tmux attach -t evaluation
```

### Option B: Run directly (stay connected)

```bash
python generate_test_set.py \
    --checkpoint Grad-TTS/logs/hungarian_dysarthria/grad_500.pt \
    --manifest test_manifest.txt \
    --output-dir generated_test_wavs \
    --length-scale 1.0 \
    --temperature 1.2 \
    --timesteps 20
```

**What you'll see:**
```
Loading checkpoint: Grad-TTS/logs/hungarian_dysarthria/grad_500.pt
✓ Model loaded from epoch 500
✓ Model moved to GPU
Loading HiFi-GAN vocoder...
✓ Vocoder loaded
Found 1016 utterances

Generating...
100%|████████████| 1016/1016 [1:23:45<00:00,  0.20it/s]

GENERATION COMPLETE
Total utterances: 1016
Successfully generated: 1016
Average RTF: 0.102
```

**Expected time:** 1-2 hours with GPU

**Check progress (if in tmux, from another terminal):**
```bash
watch -n 10 'ls generated_test_wavs/ | wc -l'
# Shows how many files generated so far
```

---

## Step 5: Compute Evaluation Metrics (30-60 min)

```bash
python evaluate_metrics.py \
    --real-dir wavs_16khz \
    --synth-dir generated_test_wavs \
    --manifest test_manifest.txt \
    --output evaluation_results.csv
```

**What you'll see:**
```
TTS INTRUSIVE QUALITY EVALUATION
Real audio directory: wavs_16khz
Synthetic audio directory: generated_test_wavs
Manifest: test_manifest.txt

Loading manifest...
Found 1016 utterances in manifest

Evaluating pairs...
100%|████████████| 1016/1016 [34:21<00:00,  0.49it/s]

OVERALL RESULTS (mean ± std)
============================
MCD         :   6.234 ±  1.423
F0_RMSE     :   0.312 ±  0.089
VUV_ERROR   :   7.850 ±  2.341
STOI        :   0.782 ±  0.054
ESTOI       :   0.698 ±  0.067
PPG_D       :     NaN ±    NaN

✓ Results saved to: evaluation_results.csv
✓ Summary saved to: evaluation_results_summary.csv
✓ Per-speaker summary saved to: evaluation_results_per_speaker.csv
```

**Expected time:** 30-60 minutes

---

## Step 6: Download Results to Local Machine

**From your local machine:**

```bash
cd ~/Documents/NewThesis

# Download all result files
scp makais@deep07:/home/makais/Thesis/NewThesis/NewThesis/evaluation_results*.csv .

# Download a few sample synthetic wavs
scp "makais@deep07:/home/makais/Thesis/NewThesis/NewThesis/generated_test_wavs/C_001_000*.wav" ./sample_generated/
```

---

## Step 7: View Results Locally

```bash
# View summary statistics
cat evaluation_results_summary.csv | column -s, -t

# View per-speaker (first 20 speakers)
head -n 21 evaluation_results_per_speaker.csv | column -s, -t

# Find best utterances (lowest MCD)
sort -t, -k2 -n evaluation_results.csv | head -n 10

# Find worst utterances (highest MCD)
sort -t, -k2 -rn evaluation_results.csv | head -n 10
```

---

## 🚨 Troubleshooting

### Issue: "CUDA out of memory"
```bash
# Reduce batch size or use CPU
export CUDA_VISIBLE_DEVICES=""
# Re-run generation
```

### Issue: "Module not found"
```bash
# Check environment activated
which python
# Should be in thesis/bin/

# Reinstall dependencies
pip install -r evaluation_requirements.txt
```

### Issue: Generation interrupted
```bash
# Just re-run - script skips existing files
python generate_test_set.py --checkpoint ... --manifest ...
```

### Issue: Some files missing in evaluation
```bash
# Check how many generated
ls generated_test_wavs/ | wc -l

# Check which are missing
python -c "
import os
manifest = [l.split('|')[0] for l in open('test_manifest.txt')]
generated = set(os.listdir('generated_test_wavs'))
missing = [f for f in manifest if os.path.basename(f) not in generated]
print(f'{len(missing)} missing files')
for f in missing[:10]: print(f)
"
```

---

## ⏱️ Total Time Estimate

| Step | Time |
|------|------|
| 1-3: Setup & verify | 5 min |
| 4: Generate test set | 1-2 hours |
| 5: Compute metrics | 30-60 min |
| 6-7: Download & analyze | 10 min |
| **TOTAL** | **~2-3 hours** |

---

## 🎯 What You Get

**Files created:**
1. `generated_test_wavs/` - 1,016 synthetic WAV files
2. `evaluation_results.csv` - Per-utterance metrics
3. `evaluation_results_summary.csv` - **Mean±std for presentation**
4. `evaluation_results_per_speaker.csv` - Per-speaker breakdown

**Ready for:**
- ✅ Monday presentation (objective metrics table)
- ✅ Thesis documentation (scientific evaluation)
- ✅ Speaker-level analysis (identify best/worst)

---

## 📊 Interpreting Results

### Good Quality Targets:
- **MCD**: < 6.0 dB (lower is better)
- **F0 RMSE**: < 0.3 log Hz (lower is better)
- **VUV Error**: < 8% (lower is better)
- **STOI**: > 0.75 (higher is better)
- **ESTOI**: > 0.70 (higher is better)

### What to Look For:
1. **Overall mean±std** - How good is the model on average?
2. **Per-speaker variance** - Which speakers work well/poorly?
3. **Correlations** - Does high MCD = low STOI? (should)
4. **Outliers** - Which utterances are worst? (error analysis)

---

## 🚀 One-Command Full Pipeline (Alternative)

```bash
# Run everything automatically
bash run_full_evaluation.sh

# In tmux (recommended)
tmux new -s evaluation
bash run_full_evaluation.sh
# Ctrl+B then D to detach
```

This script:
1. Checks dependencies
2. Generates test set (with resume capability)
3. Computes metrics
4. Shows summary table

---

## ✅ Success Criteria

You're done when you have:
- [x] 1,016 files in `generated_test_wavs/`
- [x] `evaluation_results.csv` with 1,016 rows
- [x] `evaluation_results_summary.csv` with mean±std
- [x] Results downloaded to local machine
- [x] Ready to add to DOKUMENTACIO.md Part 2

---

**Questions? Check:** `EVALUATION_README.md` for detailed explanations
