#!/usr/bin/env python3
"""
TTS Intrusive Quality Evaluation Metrics

Computes intrusive metrics on real vs. synthetic speech pairs:
- MCD (Mel-Cepstral Distortion) with DTW
- F0 RMSE and VUV (Voiced/Unvoiced) error
- P-STOI / ESTOI (intelligibility)
- PPG-D (Phonetic Posteriorgram Distance) - novelty metric

Usage:
    python evaluate_metrics.py \
        --real-dir wavs_16khz \
        --synth-dir generated_test_wavs \
        --manifest test_manifest.txt \
        --output results.csv
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
import soundfile as sf
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr

# For F0 extraction
import parselmouth
from parselmouth.praat import call

# For intelligibility metrics
try:
    from pystoi import stoi
    STOI_AVAILABLE = True
except ImportError:
    print("Warning: pystoi not installed. P-STOI/ESTOI will be skipped.")
    STOI_AVAILABLE = False

# For PPG-D (requires ESPnet or similar)
try:
    import torch
    PPG_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not available. PPG-D will be skipped.")
    PPG_AVAILABLE = False


def extract_mel_cepstrum(audio, sr, n_mfcc=13, n_fft=1024, hop_length=256):
    """
    Extract Mel-Cepstral coefficients.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_mfcc: Number of MFCCs
        
    Returns:
        mfcc: Mel-cepstral coefficients [n_mfcc, n_frames]
    """
    mfcc = librosa.feature.mfcc(
        y=audio, 
        sr=sr, 
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )
    return mfcc.T  # [n_frames, n_mfcc]


def compute_mcd_dtw(real_mfcc, synth_mfcc):
    """
    Compute Mel-Cepstral Distortion with DTW alignment.
    
    MCD = (10/ln(10)) * sqrt(2 * sum((c1 - c2)^2))
    
    Args:
        real_mfcc: Real audio MFCCs [T1, n_mfcc]
        synth_mfcc: Synthetic audio MFCCs [T2, n_mfcc]
        
    Returns:
        mcd: Mel-Cepstral Distortion (dB)
    """
    # DTW alignment
    distance, path = fastdtw(real_mfcc, synth_mfcc, dist=euclidean)
    
    # Compute MCD on aligned frames
    mcd_values = []
    for i, j in path:
        diff = real_mfcc[i] - synth_mfcc[j]
        mcd_frame = np.sqrt(2 * np.sum(diff ** 2))
        mcd_values.append(mcd_frame)
    
    # Convert to dB
    mcd = (10.0 / np.log(10.0)) * np.mean(mcd_values)
    return mcd


def extract_f0_vuv(audio, sr, f0_min=75, f0_max=600):
    """
    Extract F0 (pitch) and VUV (voiced/unvoiced) flags using Praat.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        f0_min: Minimum F0 (Hz)
        f0_max: Maximum F0 (Hz)
        
    Returns:
        f0: F0 contour (Hz), 0 for unvoiced frames
        vuv: Voiced/unvoiced flags (1=voiced, 0=unvoiced)
    """
    # Create Parselmouth Sound object
    sound = parselmouth.Sound(audio, sampling_frequency=sr)
    
    # Extract pitch
    pitch = call(sound, "To Pitch", 0.0, f0_min, f0_max)
    
    # Get F0 values at regular intervals
    time_step = 0.01  # 10ms
    f0_values = []
    vuv_values = []
    
    for t in np.arange(0, sound.duration, time_step):
        f0 = call(pitch, "Get value at time", t, "Hertz", "Linear")
        if f0 is not None and not np.isnan(f0):
            f0_values.append(f0)
            vuv_values.append(1)  # Voiced
        else:
            f0_values.append(0)
            vuv_values.append(0)  # Unvoiced
    
    return np.array(f0_values), np.array(vuv_values)


def compute_f0_metrics(real_f0, real_vuv, synth_f0, synth_vuv):
    """
    Compute F0 RMSE (voiced frames only) and VUV error rate.
    
    Args:
        real_f0: Real F0 contour
        real_vuv: Real VUV flags
        synth_f0: Synthetic F0 contour
        synth_vuv: Synthetic VUV flags
        
    Returns:
        f0_rmse: RMSE of log(F0) on voiced frames (log Hz)
        vuv_error: VUV decision error rate (%)
    """
    # Align lengths (take shorter)
    min_len = min(len(real_f0), len(synth_f0))
    real_f0 = real_f0[:min_len]
    real_vuv = real_vuv[:min_len]
    synth_f0 = synth_f0[:min_len]
    synth_vuv = synth_vuv[:min_len]
    
    # F0 RMSE on voiced frames
    voiced_mask = (real_vuv == 1) & (synth_vuv == 1) & (real_f0 > 0) & (synth_f0 > 0)
    if np.sum(voiced_mask) > 0:
        log_f0_real = np.log(real_f0[voiced_mask] + 1e-8)
        log_f0_synth = np.log(synth_f0[voiced_mask] + 1e-8)
        f0_rmse = np.sqrt(np.mean((log_f0_real - log_f0_synth) ** 2))
    else:
        f0_rmse = np.nan
    
    # VUV error rate
    vuv_errors = np.sum(real_vuv != synth_vuv)
    vuv_error = 100.0 * vuv_errors / len(real_vuv)
    
    return f0_rmse, vuv_error


def compute_stoi_metrics(real_audio, synth_audio, sr):
    """
    Compute P-STOI and ESTOI intelligibility metrics.
    
    Args:
        real_audio: Real audio signal
        synth_audio: Synthetic audio signal
        sr: Sample rate (must be 10kHz or 16kHz for STOI)
        
    Returns:
        stoi_score: Short-Time Objective Intelligibility
        estoi_score: Extended STOI
    """
    if not STOI_AVAILABLE:
        return np.nan, np.nan
    
    # Resample to 10kHz if needed (STOI requirement)
    if sr != 10000:
        real_audio_10k = librosa.resample(real_audio, orig_sr=sr, target_sr=10000)
        synth_audio_10k = librosa.resample(synth_audio, orig_sr=sr, target_sr=10000)
        sr_stoi = 10000
    else:
        real_audio_10k = real_audio
        synth_audio_10k = synth_audio
        sr_stoi = sr
    
    # Ensure same length
    min_len = min(len(real_audio_10k), len(synth_audio_10k))
    real_audio_10k = real_audio_10k[:min_len]
    synth_audio_10k = synth_audio_10k[:min_len]
    
    # Compute STOI
    try:
        stoi_score = stoi(real_audio_10k, synth_audio_10k, sr_stoi, extended=False)
        estoi_score = stoi(real_audio_10k, synth_audio_10k, sr_stoi, extended=True)
    except Exception as e:
        print(f"STOI computation failed: {e}")
        stoi_score = np.nan
        estoi_score = np.nan
    
    return stoi_score, estoi_score


def compute_ppg_distance(real_audio, synth_audio, sr):
    """
    Compute Phonetic Posteriorgram Distance (PPG-D).
    
    This is a placeholder - requires a trained phoneme recognizer.
    In practice, use ESPnet ASR model or similar.
    
    Args:
        real_audio: Real audio signal
        synth_audio: Synthetic audio signal
        sr: Sample rate
        
    Returns:
        ppg_d: PPG distance (lower is better)
    """
    # TODO: Implement with actual phoneme recognizer
    # For now, return placeholder
    print("Warning: PPG-D not implemented yet. Returning NaN.")
    return np.nan


def evaluate_pair(real_path, synth_path, sr=16000):
    """
    Evaluate one real-synthetic pair with all metrics.
    
    Args:
        real_path: Path to real audio
        synth_path: Path to synthetic audio
        sr: Sample rate
        
    Returns:
        metrics: Dictionary of metric values
    """
    # Load audio
    real_audio, _ = librosa.load(real_path, sr=sr)
    synth_audio, _ = librosa.load(synth_path, sr=sr)
    
    metrics = {}
    
    # MCD with DTW
    try:
        real_mfcc = extract_mel_cepstrum(real_audio, sr)
        synth_mfcc = extract_mel_cepstrum(synth_audio, sr)
        metrics['mcd'] = compute_mcd_dtw(real_mfcc, synth_mfcc)
    except Exception as e:
        print(f"MCD failed for {real_path}: {e}")
        metrics['mcd'] = np.nan
    
    # F0 and VUV
    try:
        real_f0, real_vuv = extract_f0_vuv(real_audio, sr)
        synth_f0, synth_vuv = extract_f0_vuv(synth_audio, sr)
        f0_rmse, vuv_error = compute_f0_metrics(real_f0, real_vuv, synth_f0, synth_vuv)
        metrics['f0_rmse'] = f0_rmse
        metrics['vuv_error'] = vuv_error
    except Exception as e:
        print(f"F0 extraction failed for {real_path}: {e}")
        metrics['f0_rmse'] = np.nan
        metrics['vuv_error'] = np.nan
    
    # STOI metrics
    try:
        stoi_score, estoi_score = compute_stoi_metrics(real_audio, synth_audio, sr)
        metrics['stoi'] = stoi_score
        metrics['estoi'] = estoi_score
    except Exception as e:
        print(f"STOI failed for {real_path}: {e}")
        metrics['stoi'] = np.nan
        metrics['estoi'] = np.nan
    
    # PPG-D (placeholder)
    metrics['ppg_d'] = compute_ppg_distance(real_audio, synth_audio, sr)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate TTS quality with intrusive metrics')
    parser.add_argument('--real-dir', type=str, required=True,
                        help='Directory with real audio files')
    parser.add_argument('--synth-dir', type=str, required=True,
                        help='Directory with synthetic audio files')
    parser.add_argument('--manifest', type=str, required=True,
                        help='Manifest file with utterance list (pipe-separated: path|text|speaker)')
    parser.add_argument('--output', type=str, default='evaluation_results.csv',
                        help='Output CSV file for results')
    parser.add_argument('--sr', type=int, default=16000,
                        help='Sample rate (default: 16000)')
    args = parser.parse_args()
    
    print('='*80)
    print('TTS INTRUSIVE QUALITY EVALUATION')
    print('='*80)
    print(f'Real audio directory: {args.real_dir}')
    print(f'Synthetic audio directory: {args.synth_dir}')
    print(f'Manifest: {args.manifest}')
    print(f'Output: {args.output}')
    print('='*80)
    
    # Load manifest
    print('\nLoading manifest...')
    
    if args.manifest.endswith('.csv'):
        # CSV format with headers: utt_id,wav,speaker,text
        import csv
        with open(args.manifest, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            lines = [(row['wav'], row['text'], row['speaker']) for row in reader]
    else:
        # Pipe-separated format: path|text|speaker
        with open(args.manifest, 'r', encoding='utf-8') as f:
            lines = [line.strip().split('|') for line in f.readlines()]
    
    print(f'Found {len(lines)} utterances in manifest')
    
    # Evaluate all pairs
    print('\nEvaluating pairs...')
    results = []
    
    for wav_path, text, speaker_id in tqdm(lines):
        # Get base filename
        basename = Path(wav_path).name
        
        # Construct paths
        real_path = os.path.join(args.real_dir, basename)
        synth_path = os.path.join(args.synth_dir, basename)
        
        # Check if both files exist
        if not os.path.exists(real_path):
            print(f"Warning: Real file not found: {real_path}")
            continue
        if not os.path.exists(synth_path):
            print(f"Warning: Synthetic file not found: {synth_path}")
            continue
        
        # Evaluate
        metrics = evaluate_pair(real_path, synth_path, sr=args.sr)
        metrics['utt_id'] = basename
        metrics['speaker_id'] = speaker_id
        metrics['text'] = text
        
        results.append(metrics)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Compute overall statistics
    print('\n' + '='*80)
    print('OVERALL RESULTS (mean ± std)')
    print('='*80)
    
    metric_names = ['mcd', 'f0_rmse', 'vuv_error', 'stoi', 'estoi', 'ppg_d']
    for metric in metric_names:
        if metric in df.columns:
            values = df[metric].dropna()
            if len(values) > 0:
                mean_val = values.mean()
                std_val = values.std()
                print(f'{metric.upper():12s}: {mean_val:7.3f} ± {std_val:6.3f}')
            else:
                print(f'{metric.upper():12s}: N/A')
    
    # Per-speaker statistics
    print('\n' + '='*80)
    print('PER-SPEAKER RESULTS')
    print('='*80)
    
    speaker_stats = df.groupby('speaker_id')[metric_names].agg(['mean', 'std', 'count'])
    print(speaker_stats.to_string())
    
    # Save results
    df.to_csv(args.output, index=False)
    print(f'\n✓ Results saved to: {args.output}')
    
    # Save summary statistics
    summary_path = args.output.replace('.csv', '_summary.csv')
    summary_df = pd.DataFrame({
        'metric': metric_names,
        'mean': [df[m].mean() for m in metric_names],
        'std': [df[m].std() for m in metric_names],
        'count': [df[m].count() for m in metric_names]
    })
    summary_df.to_csv(summary_path, index=False)
    print(f'✓ Summary saved to: {summary_path}')
    
    # Save per-speaker summary
    speaker_summary_path = args.output.replace('.csv', '_per_speaker.csv')
    speaker_stats.to_csv(speaker_summary_path)
    print(f'✓ Per-speaker summary saved to: {speaker_summary_path}')
    
    print('\n' + '='*80)
    print('EVALUATION COMPLETE')
    print('='*80)


if __name__ == '__main__':
    main()
