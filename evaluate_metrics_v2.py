#!/usr/bin/env python3
"""
TTS Intrusive Quality Evaluation - Clean Implementation

Follows best practices from literature:
- MCD with DTW (Kubichek constant)
- L-F0 RMSE + VUV error (Praat)
- P-STOI / ESTOI
- PPG-D (Wav2Vec2 CTC)

Usage:
    python evaluate_metrics_v2.py \
        --real-dir wavs_16khz \
        --synth-dir generated_test_wavs \
        --manifest test_manifest.txt \
        --output results_v2.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import librosa
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.signal import butter, lfilter

# F0 extraction
import parselmouth
from parselmouth.praat import call

# Intelligibility
try:
    from pystoi import stoi
    STOI_AVAILABLE = True
except ImportError:
    print("Warning: pystoi not installed. STOI metrics will be NaN.")
    STOI_AVAILABLE = False

# PPG extraction
try:
    import torch
    import torch.nn.functional as F
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
    PPG_AVAILABLE = True
except ImportError:
    print("Warning: transformers/torch not available. PPG-D will be NaN.")
    PPG_AVAILABLE = False


# ============================================================================
# 0) COMMON PREPROCESSING
# ============================================================================

def load_wav(path, sr=16000):
    """Load audio as mono float32."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def peak_normalize(y, eps=1e-8):
    """Peak normalize to [-1, 1]."""
    peak = np.max(np.abs(y))
    if peak < eps:
        return y
    return y / peak


def trim_both(real, synth, top_db=30):
    """
    Trim silence from both signals and truncate to same length.
    
    This ensures fair comparison without global alignment artifacts.
    """
    real_t, _ = librosa.effects.trim(real, top_db=top_db)
    synth_t, _ = librosa.effects.trim(synth, top_db=top_db)
    
    # Truncate to shorter length
    T = min(len(real_t), len(synth_t))
    return real_t[:T], synth_t[:T]


# ============================================================================
# 1) MCD (Mel-Cepstral Distortion) with DTW
# ============================================================================

_MCD_CONST = 10.0 * np.sqrt(2.0) / np.log(10.0)  # ≈ 6.14185


def mfcc_for_mcd(y, sr=16000, n_mfcc=24, n_fft=1024, hop_length=256, 
                 win_length=1024, fmin=0, fmax=8000):
    """
    Extract MFCC for MCD computation (C0 removed).
    
    Standard config:
    - 24 coefficients (+ C0 which we drop)
    - HTK=False (librosa default)
    - norm='ortho' (energy normalization)
    """
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc+1,  # +1 for C0
        n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        fmin=fmin, fmax=fmax, htk=False, norm='ortho'
    )
    mfcc = mfcc[1:, :].T  # (T, n_mfcc), drop C0
    return mfcc.astype(np.float32)


def mcd_dtw(real_y, synth_y, sr=16000, cmvn=True):
    """
    Compute MCD with DTW alignment.
    
    Args:
        real_y, synth_y: Audio signals (float32, mono)
        sr: Sample rate
        cmvn: Apply cepstral mean-variance normalization
        
    Returns:
        mcd: Mel-Cepstral Distortion in dB
    """
    A = mfcc_for_mcd(real_y, sr=sr)
    B = mfcc_for_mcd(synth_y, sr=sr)
    
    if A.shape[0] == 0 or B.shape[0] == 0:
        return float('nan')
    
    # Optional: CMVN (helps with global gain differences)
    if cmvn:
        A = (A - A.mean(axis=0)) / (A.std(axis=0) + 1e-8)
        B = (B - B.mean(axis=0)) / (B.std(axis=0) + 1e-8)
    
    # DTW alignment
    _, path = fastdtw(A, B, dist=euclidean)
    
    # Frame-wise RMSE along DTW path
    frame_rmse = []
    for i, j in path:
        diff = A[i] - B[j]
        frame_rmse.append(np.sqrt(np.mean(diff * diff)))
    
    # Apply Kubichek constant
    mcd = _MCD_CONST * float(np.mean(frame_rmse))
    return mcd


# ============================================================================
# 2) L-F0 RMSE + VUV Error (Praat)
# ============================================================================

def f0_vuv_praat(y, sr=16000, f0_min=80, f0_max=600, frame_step=0.01):
    """
    Extract F0 and V/UV decisions using Praat.
    
    Args:
        y: Audio signal
        sr: Sample rate
        f0_min, f0_max: Pitch range (Hz) - adjust for dysarthria
        frame_step: Analysis frame step (seconds)
        
    Returns:
        f0: F0 contour (0 = unvoiced)
        vuv: Voicing decisions (0=unvoiced, 1=voiced)
    """
    snd = parselmouth.Sound(y, sampling_frequency=sr)
    pitch = call(snd, "To Pitch", 0.0, f0_min, f0_max)
    
    times = np.arange(0, snd.duration, frame_step, dtype=np.float64)
    f0 = np.zeros(len(times), dtype=np.float32)
    vuv = np.zeros(len(times), dtype=np.int32)
    
    for k, t in enumerate(times):
        val = call(pitch, "Get value at time", float(t), "Hertz", "Linear")
        if val is not None and not np.isnan(val) and val > 0:
            f0[k] = float(val)
            vuv[k] = 1
    
    return f0, vuv


def lf0_rmse_and_vuv(real_y, synth_y, sr=16000, f0_min=80, f0_max=600):
    """
    Compute log-F0 RMSE and VUV error.
    
    Returns:
        lf0_rmse: Log-F0 RMSE on jointly voiced frames
        vuv_err: VUV error rate (%)
        n_joint: Number of jointly voiced frames
    """
    r_f0, r_vuv = f0_vuv_praat(real_y, sr=sr, f0_min=f0_min, f0_max=f0_max)
    s_f0, s_vuv = f0_vuv_praat(synth_y, sr=sr, f0_min=f0_min, f0_max=f0_max)
    
    # Align lengths
    T = min(len(r_f0), len(s_f0))
    r_f0, r_vuv = r_f0[:T], r_vuv[:T]
    s_f0, s_vuv = s_f0[:T], s_vuv[:T]
    
    if T == 0:
        return float('nan'), float('nan'), 0
    
    # Jointly voiced frames
    joint = (r_vuv == 1) & (s_vuv == 1) & (r_f0 > 0) & (s_f0 > 0)
    
    if np.any(joint):
        lf0_r = np.log(r_f0[joint] + 1e-8)
        lf0_s = np.log(s_f0[joint] + 1e-8)
        lf0_rmse = float(np.sqrt(np.mean((lf0_r - lf0_s) ** 2)))
    else:
        lf0_rmse = float('nan')
    
    # VUV error (percentage of disagreements)
    vuv_err = float(100.0 * np.mean(r_vuv != s_vuv))
    
    return lf0_rmse, vuv_err, int(joint.sum())


# ============================================================================
# 3) P-STOI and ESTOI
# ============================================================================


def dc_remove(x):
    """Remove DC offset."""
    return x - np.mean(x)


def peak_normalize_target(x, target=0.95, eps=1e-8):
    """Peak normalize to target amplitude."""
    m = np.max(np.abs(x)) + eps
    return x * (target / m)


def bandpass_300_3400(x, sr, low=300, high=3400, order=4):
    """
    Bandpass filter 300-3400 Hz (telephone band).
    
    Critical for STOI: removes DC, low-freq rumble, and high-freq artifacts.
    """
    nyq = 0.5 * sr
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return lfilter(b, a, x)


def lowpass_filter(x, sr, cutoff=7800, order=6):
    """
    Lowpass filter to remove high-frequency artifacts.
    
    De-esser style filter to reduce vocoder sibilance.
    """
    nyq = 0.5 * sr
    b, a = butter(order, cutoff/nyq, btype='low')
    return lfilter(b, a, x)


def stoi_preprocess(x, sr):
    """
    STOI-specific preprocessing pipeline.
    
    1. DC removal
    2. Peak normalization to 0.95
    3. Bandpass 300-3400 Hz (telephone band)
    """
    x = dc_remove(x)
    x = peak_normalize_target(x, target=0.95)
    x = bandpass_300_3400(x, sr)
    return x.astype(np.float32)


def stoi_estoi(real_y, synth_y, sr=16000):
    """
    Compute STOI and extended STOI with proper preprocessing.
    
    Critical preprocessing steps:
    - DC removal
    - Peak normalization
    - Bandpass filtering (300-3400 Hz)
    - Separate trimming for real and synth
    """
    if not STOI_AVAILABLE:
        return float('nan'), float('nan')
    
    if len(real_y) == 0 or len(synth_y) == 0:
        return float('nan'), float('nan')
    
    # Apply lowpass to synth to reduce vocoder artifacts
    synth_y = lowpass_filter(synth_y, sr, cutoff=7800, order=6)
    
    # Trim separately (critical: don't use same indices!)
    real_trimmed, _ = librosa.effects.trim(real_y, top_db=30, 
                                          frame_length=2048, hop_length=512)
    synth_trimmed, _ = librosa.effects.trim(synth_y, top_db=30,
                                           frame_length=2048, hop_length=512)
    
    # Truncate to same length
    T = min(len(real_trimmed), len(synth_trimmed))
    if T < 1600:  # < 0.1 sec
        return float('nan'), float('nan')
    
    real_aligned = real_trimmed[:T]
    synth_aligned = synth_trimmed[:T]
    
    # STOI preprocessing
    real_proc = stoi_preprocess(real_aligned, sr)
    synth_proc = stoi_preprocess(synth_aligned, sr)
    
    try:
        s1 = stoi(real_proc, synth_proc, sr, extended=False)
        s2 = stoi(real_proc, synth_proc, sr, extended=True)
        return float(s1), float(s2)
    except Exception as e:
        print(f"STOI computation failed: {e}")
        return float('nan'), float('nan')


# ============================================================================
# 4) PPG-D (Phonetic Posteriorgram Distance)
# ============================================================================

class PPGExtractor:
    """Extract PPG using pretrained Wav2Vec2 CTC model."""
    
    def __init__(self, model_name="facebook/wav2vec2-base-960h", device="cpu"):
        if not PPG_AVAILABLE:
            raise RuntimeError("PyTorch/transformers not available")
        
        self.device = device
        self.proc = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
        self.model.eval()
    
    @torch.no_grad()
    def ppg(self, y, sr=16000):
        """
        Extract phonetic posteriorgram.
        
        Args:
            y: Audio signal (float32, mono, [-1, 1])
            sr: Sample rate (must be 16000 for wav2vec2)
            
        Returns:
            ppg: Posteriorgram matrix (T, vocab_size)
        """
        if sr != 16000:
            raise ValueError("Wav2Vec2 requires 16kHz audio")
        
        inputs = self.proc(y, sampling_rate=sr, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(self.device)
        logits = self.model(input_values).logits[0]  # (T, vocab)
        post = F.softmax(logits, dim=-1).cpu().numpy()
        return post.astype(np.float32)


def ppg_distance_dtw(ppg_a, ppg_b):
    """
    Compute PPG distance with DTW alignment.
    
    Args:
        ppg_a, ppg_b: Posteriorgrams (T, V)
        
    Returns:
        distance: Mean euclidean distance along DTW path
    """
    if ppg_a.shape[0] == 0 or ppg_b.shape[0] == 0:
        return float('nan')
    
    _, path = fastdtw(ppg_a, ppg_b, dist=euclidean)
    distances = [euclidean(ppg_a[i], ppg_b[j]) for i, j in path]
    return float(np.mean(distances))


# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================

def evaluate_pair(real_path, synth_path, ppg_extractor=None):
    """
    Evaluate one real-synthetic pair with all metrics.
    
    Returns:
        dict with all metric values
    """
    # Load and preprocess
    real = load_wav(real_path)
    synth = load_wav(synth_path)
    
    # Peak normalize
    real = peak_normalize(real)
    synth = peak_normalize(synth)
    
    # Trim silence and align lengths
    real_trim, synth_trim = trim_both(real, synth, top_db=30)
    
    if len(real_trim) < 1600 or len(synth_trim) < 1600:  # < 0.1 sec
        return {
            'mcd': float('nan'),
            'lf0_rmse': float('nan'),
            'vuv_error': float('nan'),
            'n_joint_voiced': 0,
            'stoi': float('nan'),
            'estoi': float('nan'),
            'ppg_d': float('nan')
        }
    
    # 1. MCD
    mcd = mcd_dtw(real_trim, synth_trim, sr=16000, cmvn=True)
    
    # 2. F0 metrics
    lf0_rmse, vuv_err, n_joint = lf0_rmse_and_vuv(real_trim, synth_trim, sr=16000)
    
    # 3. STOI
    stoi_val, estoi_val = stoi_estoi(real_trim, synth_trim, sr=16000)
    
    # 4. PPG-D
    ppg_d = float('nan')
    if ppg_extractor is not None:
        try:
            ppg_real = ppg_extractor.ppg(real_trim, sr=16000)
            ppg_synth = ppg_extractor.ppg(synth_trim, sr=16000)
            ppg_d = ppg_distance_dtw(ppg_real, ppg_synth)
        except Exception as e:
            print(f"PPG-D failed: {e}")
    
    return {
        'mcd': mcd,
        'lf0_rmse': lf0_rmse,
        'vuv_error': vuv_err,
        'n_joint_voiced': n_joint,
        'stoi': stoi_val,
        'estoi': estoi_val,
        'ppg_d': ppg_d
    }


def load_manifest(manifest_path):
    """
    Load test manifest (CSV format: utt_id,wav,speaker,text).
    
    Returns list of (audio_filename, text, speaker_id).
    """
    import csv
    pairs = []
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            wav_path = row['wav']
            filename = os.path.basename(wav_path)
            text = row.get('text', '')
            speaker = row.get('speaker', 'unknown')
            pairs.append((filename, text, speaker))
    
    return pairs


def main():
    parser = argparse.ArgumentParser(description='Intrusive TTS Evaluation (Clean)')
    parser.add_argument('--real-dir', type=str, required=True,
                       help='Directory with real audio files')
    parser.add_argument('--synth-dir', type=str, required=True,
                       help='Directory with synthetic audio files')
    parser.add_argument('--manifest', type=str, required=True,
                       help='Test manifest file (audio_path|text|speaker_id)')
    parser.add_argument('--output', type=str, default='evaluation_results_v2.csv',
                       help='Output CSV file')
    parser.add_argument('--use-ppg', action='store_true',
                       help='Enable PPG-D computation (requires GPU, slow)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for PPG extraction (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Load manifest
    print(f"Loading manifest: {args.manifest}")
    pairs = load_manifest(args.manifest)
    print(f"Found {len(pairs)} test pairs")
    
    # Initialize PPG extractor if requested
    ppg_extractor = None
    if args.use_ppg and PPG_AVAILABLE:
        print(f"Initializing PPG extractor on {args.device}...")
        ppg_extractor = PPGExtractor(device=args.device)
        print("✓ PPG extractor ready")
    elif args.use_ppg:
        print("Warning: --use-ppg requested but dependencies not available")
    
    # Evaluate all pairs
    results = []
    
    for filename, text, speaker in tqdm(pairs, desc="Evaluating"):
        real_path = os.path.join(args.real_dir, filename)
        synth_path = os.path.join(args.synth_dir, filename)
        
        # Check if files exist
        if not os.path.exists(real_path):
            print(f"Warning: Real file not found: {real_path}")
            continue
        if not os.path.exists(synth_path):
            print(f"Warning: Synth file not found: {synth_path}")
            continue
        
        # Evaluate
        metrics = evaluate_pair(real_path, synth_path, ppg_extractor)
        
        # Store results
        results.append({
            'filename': filename,
            'speaker': speaker,
            'text': text,
            **metrics
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Compute summary statistics
    print("\n" + "="*80)
    print("EVALUATION RESULTS SUMMARY")
    print("="*80)
    
    metrics_to_summarize = ['mcd', 'lf0_rmse', 'vuv_error', 'stoi', 'estoi', 'ppg_d']
    
    for metric in metrics_to_summarize:
        if metric in df.columns:
            values = df[metric].dropna()
            if len(values) > 0:
                mean = values.mean()
                std = values.std()
                print(f"{metric.upper():15s}: {mean:8.4f} ± {std:8.4f} (n={len(values)})")
            else:
                print(f"{metric.upper():15s}: No valid values")
    
    print("="*80)
    
    # Save results
    df.to_csv(args.output, index=False)
    print(f"\n✓ Results saved to: {args.output}")
    
    # Per-speaker summary
    if len(df['speaker'].unique()) > 1:
        speaker_summary = df.groupby('speaker')[metrics_to_summarize].mean()
        output_speaker = args.output.replace('.csv', '_per_speaker.csv')
        speaker_summary.to_csv(output_speaker)
        print(f"✓ Per-speaker results saved to: {output_speaker}")


if __name__ == '__main__':
    main()
