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

# For mel-spectrogram (matching training pipeline)
try:
    from speechbrain.lobes.features import Fbank
    from speechbrain.processing.features import STFT
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    print("Warning: SpeechBrain not available. Using librosa for mel-spectrogram.")
    SPEECHBRAIN_AVAILABLE = False

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


def trim_silence(audio, sr, top_db=30, frame_length=2048, hop_length=512):
    """
    Trim leading and trailing silence from audio using energy-based VAD.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        top_db: Threshold in dB below reference to consider as silence
        
    Returns:
        trimmed_audio: Audio with silence removed
    """
    trimmed, _ = librosa.effects.trim(
        audio, 
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length
    )
    return trimmed


def estimate_global_delay(real_audio, synth_audio, sr, max_shift_sec=0.5):
    """
    Estimate global time delay using ROBUST RMS envelope with normalized cross-correlation.
    
    CRITICAL: This is the paper-standard approach to handle TTS onset/offset misalignment.
    
    Robust method:
    1. Compute RMS energy envelope (NOT abs - more stable for dysarthria)
    2. Log-scale transform (stabilizes dynamic range)
    3. Z-score normalization (mean=0, std=1)
    4. Normalized cross-correlation
    5. Peak quality check (reject weak/ambiguous alignments)
    
    Args:
        real_audio: Real audio signal (already trimmed to speech)
        synth_audio: Synthetic audio signal (already trimmed to speech)
        sr: Sample rate
        max_shift_sec: Maximum allowed shift in seconds (default 0.5s = ±500ms)
        
    Returns:
        delay_samples: Optimal delay in samples (positive = synth is delayed)
        correlation_strength: Peak quality metric (z-score of peak)
        is_reliable: Boolean flag (True if alignment is reliable)
    """
    # Compute RMS energy envelope (25ms frame, 10ms hop)
    frame_length = int(0.025 * sr)  # 25ms = 400 samples @ 16kHz
    hop_length = int(0.01 * sr)     # 10ms = 160 samples @ 16kHz
    
    # RMS energy for real signal
    real_rms = librosa.feature.rms(y=real_audio, frame_length=frame_length, hop_length=hop_length)[0]
    # RMS energy for synth signal
    synth_rms = librosa.feature.rms(y=synth_audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Log-scale transform (stabilizes dynamic range, critical for dysarthria)
    eps = 1e-8
    real_log_energy = np.log(real_rms + eps)
    synth_log_energy = np.log(synth_rms + eps)
    
    # Z-score normalization (mean=0, std=1) with guards
    real_mean = np.mean(real_log_energy)
    real_std = np.std(real_log_energy)
    synth_mean = np.mean(synth_log_energy)
    synth_std = np.std(synth_log_energy)
    
    # Guard against flat signals (std too small)
    if real_std < 0.1 or synth_std < 0.1:
        # Signal is too flat, alignment unreliable
        return 0, 0.0, False
    
    real_env = (real_log_energy - real_mean) / real_std
    synth_env = (synth_log_energy - synth_mean) / synth_std
    
    # Normalized cross-correlation (NCC)
    # Normalize by signal energy to make correlation scale-invariant
    real_norm = np.linalg.norm(real_env)
    synth_norm = np.linalg.norm(synth_env)
    
    if real_norm < eps or synth_norm < eps:
        return 0, 0.0, False
    
    correlation = np.correlate(real_env, synth_env, mode='full') / (real_norm * synth_norm + eps)
    
    # Find peak within allowed range (±max_shift_sec)
    max_shift_frames = int(max_shift_sec * sr / hop_length)
    center = len(synth_env) - 1  # Zero-lag position
    search_start = max(0, center - max_shift_frames)
    search_end = min(len(correlation), center + max_shift_frames + 1)
    
    # Extract search region
    search_region = correlation[search_start:search_end]
    
    # Find best peak
    peak_idx = np.argmax(search_region)
    peak_value = search_region[peak_idx]
    
    # Compute peak quality (z-score of peak within search region)
    search_mean = np.mean(search_region)
    search_std = np.std(search_region)
    
    if search_std < eps:
        # Flat correlation, no clear peak
        return 0, 0.0, False
    
    peak_z_score = (peak_value - search_mean) / search_std
    
    # Additional check: peak margin (distance to second-best peak)
    search_region_sorted = np.sort(search_region)
    if len(search_region_sorted) > 1:
        second_peak = search_region_sorted[-2]
        peak_margin = (peak_value - second_peak) / (search_std + eps)
    else:
        peak_margin = 0.0
    
    # Reliability criteria:
    # - Peak z-score > 3.0 (peak is 3 std above mean)
    # - Peak margin > 1.0 (clear winner, not ambiguous)
    is_reliable = (peak_z_score > 3.0) and (peak_margin > 1.0)
    
    # Convert frame delay to sample delay
    delay_frames = peak_idx - max_shift_frames
    delay_samples = delay_frames * hop_length
    
    return delay_samples, peak_z_score, is_reliable


def apply_delay_correction(real_audio, synth_audio, delay_samples):
    """
    Apply delay correction to align real and synth audio.
    
    Args:
        real_audio: Real audio signal
        synth_audio: Synthetic audio signal
        delay_samples: Delay in samples (from estimate_global_delay)
        
    Returns:
        real_aligned, synth_aligned: Time-aligned audio signals
    """
    if delay_samples > 0:
        # Synth is delayed, shift it left (trim start)
        synth_audio = synth_audio[delay_samples:]
        real_audio = real_audio[:len(synth_audio)]
    elif delay_samples < 0:
        # Synth is ahead, shift it right (trim start of real)
        real_audio = real_audio[-delay_samples:]
        synth_audio = synth_audio[:len(real_audio)]
    
    # Final length alignment
    min_len = min(len(real_audio), len(synth_audio))
    return real_audio[:min_len], synth_audio[:min_len]


def align_audio_length(audio1, audio2):
    """
    Align two audio signals to the same length by truncating to minimum.
    
    Args:
        audio1: First audio signal
        audio2: Second audio signal
        
    Returns:
        audio1_aligned, audio2_aligned: Both signals with same length
    """
    min_len = min(len(audio1), len(audio2))
    return audio1[:min_len], audio2[:min_len]


def extract_mel_cepstrum(audio, sr, n_mfcc=24, n_fft=1024, hop_length=256, win_length=1024, f_min=0, f_max=8000):
    """
    Extract Mel-Cepstral Coefficients (MCEP) for MCD computation.
    
    CRITICAL: MCD (Mel-Cepstral Distortion) is DEFINED on mel-cepstral coefficients,
    NOT log mel-spectrogram bins. Using MFCC as standard approximation to MCEP.
    
    Standard TTS papers use 24-25 MFCCs (excluding C0 = energy) for MCD computation.
    
    Reference: Kubichek (1993), "Mel-Cepstral Distance Measure for Objective 
    Speech Quality Assessment"
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients (default 24, C0 will be excluded)
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        f_min: Minimum frequency
        f_max: Maximum frequency
        
    Returns:
        mfcc: MFCC coefficients [n_frames, n_mfcc] (C0 excluded)
    """
    # Extract MFCCs (mel-cepstral coefficients)
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc + 1,  # +1 because we'll exclude C0 (energy)
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        fmin=f_min,
        fmax=f_max,
        norm='ortho',  # Standard DCT-II orthonormal normalization
        htk=False
    )
    
    # Exclude C0 (energy coefficient) - standard practice for MCD
    mfcc = mfcc[1:, :]  # Now shape: (n_mfcc, n_frames)
    
    # Transpose to [n_frames, n_mfcc]
    mfcc = mfcc.T
    
    return mfcc


def compute_mcd_dtw(real_mfcc, synth_mfcc, use_cmvn=True):
    """
    Compute Mel-Cepstral Distortion (MCD) with DTW alignment.
    
    Standard MCD formula (Kubichek 1993):
    MCD = (10 * sqrt(2) / ln(10)) * sqrt(sum_{k=1}^{K} (c_k^real - c_k^synth)^2 / K)
        ≈ 6.141 * sqrt(mean((c_real - c_synth)^2))
    
    where c_k are mel-cepstral coefficients (MFCC, excluding C0).
    
    Args:
        real_mfcc: Real audio MFCC [T1, n_mfcc]
        synth_mfcc: Synthetic audio MFCC [T2, n_mfcc]
        use_cmvn: Whether to apply Cepstral Mean-Variance Normalization before DTW
        
    Returns:
        mcd: Mel-Cepstral Distortion (dB)
    """
    # Apply CMVN (per-utterance normalization) to reduce speaker/channel variance
    if use_cmvn:
        # Zero-mean, unit-variance normalization
        real_mfcc = (real_mfcc - np.mean(real_mfcc, axis=0)) / (np.std(real_mfcc, axis=0) + 1e-8)
        synth_mfcc = (synth_mfcc - np.mean(synth_mfcc, axis=0)) / (np.std(synth_mfcc, axis=0) + 1e-8)
    
    # DTW alignment
    distance, path = fastdtw(real_mfcc, synth_mfcc, dist=euclidean)
    
    # Compute MCD on aligned frames
    # Formula: 6.141 * sqrt(mean((c_real - c_synth)^2))
    mcd_values = []
    for i, j in path:
        diff = real_mfcc[i] - synth_mfcc[j]
        # RMSE across cepstral coefficients
        frame_mcd = np.sqrt(np.mean(diff ** 2))
        mcd_values.append(frame_mcd)
    
    # Average MCD over all aligned frames, apply MCD constant
    mcd = np.mean(mcd_values) * 10 * np.sqrt(2) / np.log(10)  # ≈ 6.141 * RMSE
    
    return mcd


def extract_f0_vuv(audio, sr, f0_min=80, f0_max=600, hop_length=256):
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
        # CRITICAL: Praat returns 0 for unvoiced, not necessarily NaN
        # Only mark as voiced if f0 is valid AND greater than 0
        if f0 is not None and not np.isnan(f0) and f0 > 0:
            f0_values.append(f0)
            vuv_values.append(1)  # Voiced
        else:
            f0_values.append(0)
            vuv_values.append(0)  # Unvoiced
    
    return np.array(f0_values), np.array(vuv_values)


def compute_f0_metrics(real_f0, real_vuv, synth_f0, synth_vuv):
    """
    Compute F0 RMSE (voiced frames only) and VUV error rate.
    
    NOTE: Returns BOTH raw F0 RMSE (Hz) and log F0 RMSE for comparison.
    
    Args:
        real_f0: Real F0 contour (Hz)
        real_vuv: Real VUV flags (1=voiced, 0=unvoiced)
        synth_f0: Synthetic F0 contour (Hz)
        synth_vuv: Synthetic VUV flags (1=voiced, 0=unvoiced)
        
    Returns:
        f0_rmse_hz: RMSE of F0 in Hz (on jointly voiced frames)
        f0_rmse_log: RMSE of log(F0) (on jointly voiced frames)
        vuv_error: VUV decision error rate (%)
        voiced_frames_real: Number of voiced frames in real audio
        voiced_frames_synth: Number of voiced frames in synthetic audio
        voiced_frames_joint: Number of jointly voiced frames
    """
    # Align lengths (take shorter)
    min_len = min(len(real_f0), len(synth_f0))
    real_f0 = real_f0[:min_len]
    real_vuv = real_vuv[:min_len]
    synth_f0 = synth_f0[:min_len]
    synth_vuv = synth_vuv[:min_len]
    
    # Count voiced frames
    voiced_frames_real = np.sum(real_vuv == 1)
    voiced_frames_synth = np.sum(synth_vuv == 1)
    
    # F0 RMSE on JOINTLY voiced frames (both signals voiced)
    voiced_mask = (real_vuv == 1) & (synth_vuv == 1) & (real_f0 > 0) & (synth_f0 > 0)
    voiced_frames_joint = np.sum(voiced_mask)
    
    if voiced_frames_joint > 0:
        # Raw F0 RMSE in Hz
        f0_rmse_hz = np.sqrt(np.mean((real_f0[voiced_mask] - synth_f0[voiced_mask]) ** 2))
        
        # Log F0 RMSE (for pitch perception)
        log_f0_real = np.log(real_f0[voiced_mask] + 1e-8)
        log_f0_synth = np.log(synth_f0[voiced_mask] + 1e-8)
        f0_rmse_log = np.sqrt(np.mean((log_f0_real - log_f0_synth) ** 2))
    else:
        f0_rmse_hz = np.nan
        f0_rmse_log = np.nan
    
    # VUV error rate (% of frames with wrong voicing decision)
    vuv_errors = np.sum(real_vuv != synth_vuv)
    vuv_error = 100.0 * vuv_errors / len(real_vuv)
    
    return f0_rmse_hz, f0_rmse_log, vuv_error, voiced_frames_real, voiced_frames_synth, voiced_frames_joint


def compute_stoi_metrics(real_audio, synth_audio, sr):
    """
    Compute P-STOI and ESTOI intelligibility metrics.
    
    CRITICAL: Both signals must be:
    - Same sample rate (10kHz or 16kHz)
    - Mono
    - Time-aligned (already VAD-trimmed and delay-corrected in evaluate_pair)
    - Same length
    - Normalized amplitude (float32, -1 to 1 range)
    
    NOTE: Global delay correction is now done in evaluate_pair() using
    envelope-based cross-correlation before calling this function.
    
    Args:
        real_audio: Real audio signal (already trimmed and aligned)
        synth_audio: Synthetic audio signal (already trimmed and aligned)
        sr: Sample rate (must be 10kHz or 16kHz for STOI)
        
    Returns:
        stoi_score: P-STOI score (0-1, higher is better)
        estoi_score: ESTOI score (0-1, higher is better)
    """
    if not STOI_AVAILABLE:
        return np.nan, np.nan
    
    if sr not in [10000, 16000]:
        print(f"Warning: STOI requires 10kHz or 16kHz, got {sr}Hz")
        return np.nan, np.nan
    
    # Ensure both signals have same length (already aligned in evaluate_pair)
    if len(real_audio) != len(synth_audio):
        print(f"Warning: Length mismatch: {len(real_audio)} vs {len(synth_audio)}")
        return np.nan, np.nan
    
    # Check for empty or very short signals
    if len(real_audio) < sr * 0.1:  # Less than 100ms
        print(f"Warning: Signal too short: {len(real_audio)} samples")
        return np.nan, np.nan
    
    try:
        # P-STOI (standard STOI)
        stoi_score = stoi(real_audio, synth_audio, sr, extended=False)
        
        # ESTOI (extended STOI, better for noisy/degraded speech)
        estoi_score = stoi(real_audio, synth_audio, sr, extended=True)
        
        return stoi_score, estoi_score
    except Exception as e:
        print(f"STOI computation failed: {e}")
        return np.nan, np.nan


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


def evaluate_pair(real_path, synth_path, sr=16000, use_vad=True):
    """
    Evaluate one real-synthetic pair with all metrics.
    
    Args:
        real_path: Path to real audio
        synth_path: Path to synthetic audio
        sr: Sample rate
        use_vad: Whether to trim silence before STOI computation
        
    Returns:
        metrics: Dictionary of metric values with diagnostics
    """
    # Load audio
    real_audio, _ = librosa.load(real_path, sr=sr)
    synth_audio, _ = librosa.load(synth_path, sr=sr)
    
    # Store original lengths for diagnostics
    orig_real_len = len(real_audio)
    orig_synth_len = len(synth_audio)
    
    metrics = {}
    
    # MCD with DTW (no trimming needed - DTW handles alignment)
    try:
        real_mel = extract_mel_cepstrum(real_audio, sr)
        synth_mel = extract_mel_cepstrum(synth_audio, sr)
        metrics['mcd'] = compute_mcd_dtw(real_mel, synth_mel)
        
        # Diagnostics: check mel-spectrogram value ranges
        metrics['mel_real_mean'] = np.mean(real_mel)
        metrics['mel_real_std'] = np.std(real_mel)
        metrics['mel_synth_mean'] = np.mean(synth_mel)
        metrics['mel_synth_std'] = np.std(synth_mel)
        
        # CRITICAL DEBUG: Check per-bin statistics
        # Log mel-spectrogram (dB) should have mean around -20 to -40 dB, std around 10-20 dB
        metrics['mel_real_min'] = np.min(real_mel)
        metrics['mel_real_max'] = np.max(real_mel)
        metrics['mel_synth_min'] = np.min(synth_mel)
        metrics['mel_synth_max'] = np.max(synth_mel)
        
        # Frame-wise Euclidean distance (before DTW)
        if real_mel.shape[0] == synth_mel.shape[0]:
            frame_dists = np.sqrt(np.sum((real_mel - synth_mel)**2, axis=1))
            metrics['mel_frame_dist_mean'] = np.mean(frame_dists)
            metrics['mel_frame_dist_std'] = np.std(frame_dists)
        else:
            metrics['mel_frame_dist_mean'] = np.nan
            metrics['mel_frame_dist_std'] = np.nan
            
    except Exception as e:
        print(f"MCD failed for {real_path}: {e}")
        metrics['mcd'] = np.nan
        metrics['mel_real_mean'] = np.nan
        metrics['mel_real_std'] = np.nan
        metrics['mel_synth_mean'] = np.nan
        metrics['mel_synth_std'] = np.nan
        metrics['mel_real_min'] = np.nan
        metrics['mel_real_max'] = np.nan
        metrics['mel_synth_min'] = np.nan
        metrics['mel_synth_max'] = np.nan
        metrics['mel_frame_dist_mean'] = np.nan
        metrics['mel_frame_dist_std'] = np.nan
    
    # F0 and VUV with detailed diagnostics
    # CRITICAL: Use the SAME aligned audio as STOI (after delay correction)
    try:
        # Get aligned audio (VAD trimmed + delay corrected)
        if use_vad:
            # Use the aligned signals from STOI section
            real_trimmed, trim_indices = librosa.effects.trim(
                real_audio,
                top_db=30,
                frame_length=2048,
                hop_length=512
            )
            start_idx = trim_indices[0]
            end_idx = trim_indices[1]
            synth_trimmed = synth_audio[start_idx:end_idx]
            
            # Apply global delay correction with peak quality check
            delay_samples, peak_z_score, is_reliable = estimate_global_delay(real_trimmed, synth_trimmed, sr, max_shift_sec=0.5)
            
            # Only apply delay if alignment is reliable
            if is_reliable:
                real_aligned_f0, synth_aligned_f0 = apply_delay_correction(real_trimmed, synth_trimmed, delay_samples)
            else:
                # Unreliable alignment, don't shift (use delay=0)
                real_aligned_f0, synth_aligned_f0 = apply_delay_correction(real_trimmed, synth_trimmed, 0)
            
            # Store correlation strength and reliability for diagnostics
            metrics['alignment_peak_z'] = peak_z_score
            metrics['alignment_reliable'] = is_reliable
        else:
            real_aligned_f0 = real_audio
            synth_aligned_f0 = synth_audio
            metrics['alignment_peak_z'] = np.nan
            metrics['alignment_reliable'] = False
        
        # Extract F0/VUV on aligned audio
        real_f0, real_vuv = extract_f0_vuv(real_aligned_f0, sr)
        synth_f0, synth_vuv = extract_f0_vuv(synth_aligned_f0, sr)
        
        f0_rmse_hz, f0_rmse_log, vuv_error, vf_real, vf_synth, vf_joint = \
            compute_f0_metrics(real_f0, real_vuv, synth_f0, synth_vuv)
        
        metrics['f0_rmse_hz'] = f0_rmse_hz  # Raw F0 RMSE in Hz
        metrics['f0_rmse_log'] = f0_rmse_log  # Log F0 RMSE
        metrics['vuv_error'] = vuv_error
        
        # Diagnostics: F0 statistics
        real_f0_voiced = real_f0[real_f0 > 0]
        synth_f0_voiced = synth_f0[synth_f0 > 0]
        
        metrics['f0_real_mean'] = np.mean(real_f0_voiced) if len(real_f0_voiced) > 0 else np.nan
        metrics['f0_real_std'] = np.std(real_f0_voiced) if len(real_f0_voiced) > 0 else np.nan
        metrics['f0_synth_mean'] = np.mean(synth_f0_voiced) if len(synth_f0_voiced) > 0 else np.nan
        metrics['f0_synth_std'] = np.std(synth_f0_voiced) if len(synth_f0_voiced) > 0 else np.nan
        
        metrics['voiced_frames_real'] = vf_real
        metrics['voiced_frames_synth'] = vf_synth
        metrics['voiced_frames_joint'] = vf_joint
        
    except Exception as e:
        print(f"F0 extraction failed for {real_path}: {e}")
        metrics['f0_rmse_hz'] = np.nan
        metrics['f0_rmse_log'] = np.nan
        metrics['vuv_error'] = np.nan
        metrics['f0_real_mean'] = np.nan
        metrics['f0_real_std'] = np.nan
        metrics['f0_synth_mean'] = np.nan
        metrics['f0_synth_std'] = np.nan
        metrics['voiced_frames_real'] = 0
        metrics['voiced_frames_synth'] = 0
        metrics['voiced_frames_joint'] = 0
    
    # STOI metrics with VAD trimming AND global delay correction
    try:
        if use_vad:
            # CRITICAL: Trim based on REAL audio only, then apply same region to SYNTH
            # This ensures both signals cover the exact same temporal region
            real_trimmed, trim_indices = librosa.effects.trim(
                real_audio,
                top_db=30,
                frame_length=2048,
                hop_length=512
            )
            
            # Apply same trim indices to synthetic audio
            start_idx = trim_indices[0]
            end_idx = trim_indices[1]
            synth_trimmed = synth_audio[start_idx:end_idx]
            
            # PAPER-STANDARD: Global delay correction using RMS envelope + normalized cross-correlation
            # This handles TTS onset/offset misalignment (critical for STOI/F0/VUV)
            delay_samples, peak_z_score, is_reliable = estimate_global_delay(real_trimmed, synth_trimmed, sr, max_shift_sec=0.5)
            
            # Only apply delay if alignment is reliable (peak_z > 3.0 AND margin > 1.0)
            if is_reliable:
                real_aligned, synth_aligned = apply_delay_correction(real_trimmed, synth_trimmed, delay_samples)
                metrics['alignment_delay_ms'] = (delay_samples / sr) * 1000.0
            else:
                # Unreliable alignment, don't shift (delay=0)
                real_aligned, synth_aligned = apply_delay_correction(real_trimmed, synth_trimmed, 0)
                metrics['alignment_delay_ms'] = 0.0
            
            # Store alignment quality metrics
            metrics['alignment_peak_z'] = peak_z_score
            metrics['alignment_reliable'] = is_reliable
            
            # Store trimmed lengths for diagnostics
            metrics['audio_real_trimmed_len'] = len(real_aligned)
            metrics['audio_synth_trimmed_len'] = len(synth_aligned)
            metrics['trim_start_idx'] = int(start_idx)
            metrics['trim_end_idx'] = int(end_idx)
        else:
            real_aligned, synth_aligned = align_audio_length(real_audio, synth_audio)
            metrics['audio_real_trimmed_len'] = len(real_aligned)
            metrics['audio_synth_trimmed_len'] = len(synth_aligned)
            metrics['trim_start_idx'] = 0
            metrics['trim_end_idx'] = len(real_audio)
        
        stoi_score, estoi_score = compute_stoi_metrics(real_aligned, synth_aligned, sr)
        metrics['stoi'] = stoi_score
        metrics['estoi'] = estoi_score
        
    except Exception as e:
        print(f"STOI failed for {real_path}: {e}")
        metrics['stoi'] = np.nan
        metrics['estoi'] = np.nan
        metrics['audio_real_trimmed_len'] = np.nan
        metrics['audio_synth_trimmed_len'] = np.nan
        metrics['trim_start_idx'] = np.nan
        metrics['trim_end_idx'] = np.nan
    
    # Original audio lengths for diagnostics
    metrics['audio_real_orig_len'] = orig_real_len
    metrics['audio_synth_orig_len'] = orig_synth_len
    
    # PPG-D (placeholder)
    metrics['ppg_d'] = compute_ppg_distance(real_audio, synth_audio, sr)
    
    return metrics


def sanity_check_metrics(audio_dir, manifest_path, output_prefix, sr=16000, mode='real_vs_real'):
    """
    Sanity check: evaluate audio against itself to verify metric implementation.
    Expected results: MCD ~ 0, STOI ~ 1, ESTOI ~ 1, F0_RMSE ~ 0, VUV_ERROR ~ 0
    
    Args:
        audio_dir: Directory with audio files
        manifest_path: Manifest file
        output_prefix: Prefix for output files
        sr: Sample rate
        mode: 'real_vs_real' or 'synth_vs_synth'
    """
    print('\n' + '='*80)
    print(f'SANITY CHECK: {mode.upper()}')
    print('='*80)
    print(f'Audio directory: {audio_dir}')
    print(f'Mode: {mode}')
    print(f'Expected: MCD~0, F0_RMSE~0, VUV_ERROR~0, STOI~1, ESTOI~1')
    print('='*80)
    
    # Load manifest
    if manifest_path.endswith('.csv'):
        import csv
        with open(manifest_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            lines = [(row['wav'], row['text'], row['speaker']) for row in reader]
    else:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            lines = [line.strip().split('|') for line in f.readlines()]
    
    print(f'\nProcessing {len(lines)} utterances...\n')
    
    results = []
    for wav_path, text, speaker_id in tqdm(lines[:50]):  # Limit to 50 samples for quick check
        basename = os.path.basename(wav_path)
        audio_path = os.path.join(audio_dir, basename)
        
        if not os.path.exists(audio_path):
            continue
        
        # Evaluate file against itself
        metrics = evaluate_pair(audio_path, audio_path, sr=sr)
        metrics['utt_id'] = basename
        metrics['speaker_id'] = speaker_id
        results.append(metrics)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Compute statistics
    print('\n' + '='*80)
    print(f'SANITY CHECK RESULTS: {mode.upper()}')
    print('='*80)
    
    metric_names = ['mcd', 'f0_rmse_hz', 'f0_rmse_log', 'vuv_error', 'stoi', 'estoi']
    for metric in metric_names:
        if metric in df.columns:
            values = df[metric].dropna()
            if len(values) > 0:
                mean_val = values.mean()
                std_val = values.std()
                min_val = values.min()
                max_val = values.max()
                print(f'{metric.upper():15s}: mean={mean_val:7.3f}, std={std_val:6.3f}, min={min_val:7.3f}, max={max_val:7.3f}')
    
    # Save results
    output_path = f'{output_prefix}_sanity_{mode}.csv'
    df.to_csv(output_path, index=False)
    print(f'\n✓ Sanity check results saved to: {output_path}')
    
    # Interpretation
    print('\n' + '='*80)
    print('INTERPRETATION:')
    print('='*80)
    mcd_mean = df['mcd'].mean() if 'mcd' in df.columns else np.nan
    f0_hz_mean = df['f0_rmse_hz'].mean() if 'f0_rmse_hz' in df.columns else np.nan
    stoi_mean = df['stoi'].mean() if 'stoi' in df.columns else None
    
    passed = True
    if not (mcd_mean < 1.0):
        passed = False
    if not (f0_hz_mean < 1.0):
        passed = False
    if stoi_mean is not None and not (stoi_mean > 0.95):
        passed = False
    
    if passed:
        print('✓ PASS: Metrics are working correctly!')
        print('  MCD is near zero and STOI is near 1 (as expected for identical files)')
    else:
        print('✗ FAIL: Metric implementation may have issues!')
        print(f'  MCD = {mcd_mean:.3f} (expected < 1.0)')
        print(f'  F0 RMSE (Hz) = {f0_hz_mean:.3f} (expected < 1.0)')
        if stoi_mean is not None:
            print(f'  STOI = {stoi_mean:.3f} (expected > 0.95)')
        print('  Check mel-spectrogram extraction and DTW alignment')
    print('='*80)


def main():
    parser = argparse.ArgumentParser(description='Evaluate TTS quality metrics')
    parser.add_argument('--real-dir', type=str,
                        help='Directory with real audio files')
    parser.add_argument('--synth-dir', type=str,
                        help='Directory with synthetic audio files')
    parser.add_argument('--manifest', type=str, required=True,
                        help='Manifest file with utterance list (pipe-separated: path|text|speaker)')
    parser.add_argument('--output', type=str, default='evaluation_results.csv',
                        help='Output CSV file for results')
    parser.add_argument('--sr', type=int, default=16000,
                        help='Sample rate (default: 16000)')
    parser.add_argument('--sanity-check', action='store_true',
                        help='Run sanity check: evaluate audio against itself')
    parser.add_argument('--sanity-mode', type=str, choices=['real', 'synth', 'both'], default='both',
                        help='Sanity check mode: real_vs_real, synth_vs_synth, or both')
    args = parser.parse_args()
    
    # Sanity check mode
    if args.sanity_check:
        output_prefix = args.output.replace('.csv', '')
        
        if args.sanity_mode in ['real', 'both']:
            if not args.real_dir:
                print("Error: --real-dir required for real_vs_real sanity check")
                return
            sanity_check_metrics(args.real_dir, args.manifest, output_prefix, 
                               sr=args.sr, mode='real_vs_real')
        
        if args.sanity_mode in ['synth', 'both']:
            if not args.synth_dir:
                print("Error: --synth-dir required for synth_vs_synth sanity check")
                return
            sanity_check_metrics(args.synth_dir, args.manifest, output_prefix, 
                               sr=args.sr, mode='synth_vs_synth')
        
        return
    
    # Normal evaluation mode
    if not args.real_dir or not args.synth_dir:
        print("Error: --real-dir and --synth-dir required for normal evaluation")
        parser.print_help()
        return
    
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
    
    # Core metrics to report
    core_metrics = ['mcd', 'f0_rmse_hz', 'f0_rmse_log', 'vuv_error', 'stoi', 'estoi', 'ppg_d']
    for metric in core_metrics:
        if metric in df.columns:
            values = df[metric].dropna()
            if len(values) > 0:
                mean_val = values.mean()
                std_val = values.std()
                print(f'{metric.upper():15s}: {mean_val:7.3f} ± {std_val:6.3f}')
            else:
                print(f'{metric.upper():15s}: N/A')
    
    # Check for suspicious MCD values
    if 'mcd' in df.columns:
        mcd_mean = df['mcd'].mean()
        if mcd_mean > 15.0:
            print(f"\n⚠️  WARNING: MCD = {mcd_mean:.1f} dB is unusually high!")
            print("   Expected range for good TTS: 4-8 dB")
            print("   Possible issues:")
            print("   - MFCC extraction mismatch (check n_mfcc, n_fft, hop_length)")
            print("   - Feature normalization (CMVN is enabled by default)")
            print("   - Vocoder quality problems")
            print("   - Model training quality issues")
            print("   Check DIAGNOSTIC STATISTICS below for MFCC value ranges\n")
        elif mcd_mean < 3.0:
            print(f"\n✓ MCD = {mcd_mean:.1f} dB is excellent! (< 3 dB suggests near-perfect match)\n")
    
    # Diagnostic statistics
    print('\n' + '='*80)
    print('DIAGNOSTIC STATISTICS')
    print('='*80)
    
    # Alignment delay statistics
    if 'alignment_delay_ms' in df.columns:
        delay_mean = df['alignment_delay_ms'].mean()
        delay_std = df['alignment_delay_ms'].std()
        print(f"Global alignment delay: {delay_mean:.1f} ± {delay_std:.1f} ms")
        if abs(delay_mean) > 100:
            print(f"⚠️  WARNING: Large average delay ({delay_mean:.0f} ms) detected!")
            print("   This suggests systematic onset timing differences between real and synth.\n")
    
    # Alignment reliability (peak quality check)
    if 'alignment_peak_z' in df.columns and 'alignment_reliable' in df.columns:
        peak_z_mean = df['alignment_peak_z'].mean()
        peak_z_std = df['alignment_peak_z'].std()
        reliable_count = df['alignment_reliable'].sum()
        total_count = len(df)
        reliable_rate = 100.0 * reliable_count / total_count if total_count > 0 else 0
        
        print(f"Alignment peak quality (z-score): {peak_z_mean:.2f} ± {peak_z_std:.2f}")
        print(f"Alignment reliability: {reliable_count}/{total_count} ({reliable_rate:.1f}%)")
        
        if reliable_rate < 50:
            print(f"⚠️  WARNING: Low reliability rate ({reliable_rate:.1f}%)!")
            print("   Most alignments are weak/ambiguous. STOI/VUV metrics are unreliable.")
            print("   Possible causes:")
            print("   - Content mismatch (real vs synth not the same utterance)")
            print("   - High tempo variability (dysarthric speech characteristic)")
            print("   - Silence/pause differences between real and synth")
            print("   Consider checking utterance pairing correctness.\n")
        elif reliable_rate < 80:
            print(f"⚠️  Moderate reliability ({reliable_rate:.1f}%). Some alignments may be unreliable.\n")
        else:
            print(f"✓ Good reliability ({reliable_rate:.1f}%). Alignments are generally stable.\n")
    
    # F0 statistics
    if 'f0_real_mean' in df.columns:
        print(f"F0 Real (Hz):  mean={df['f0_real_mean'].mean():.1f} ± {df['f0_real_std'].mean():.1f}")
    if 'f0_synth_mean' in df.columns:
        print(f"F0 Synth (Hz): mean={df['f0_synth_mean'].mean():.1f} ± {df['f0_synth_std'].mean():.1f}")
    if 'voiced_frames_real' in df.columns:
        print(f"Voiced frames: real={df['voiced_frames_real'].mean():.0f}, " + \
              f"synth={df['voiced_frames_synth'].mean():.0f}, " + \
              f"joint={df['voiced_frames_joint'].mean():.0f}")
        
        # Check if VUV alignment is problematic
        joint_ratio = df['voiced_frames_joint'].mean() / max(df['voiced_frames_real'].mean(), 1)
        if joint_ratio < 0.6:
            print(f"\n⚠️  WARNING: Joint voiced frames ({joint_ratio:.1%}) is low!")
            print("   This suggests temporal misalignment between real and synthetic audio.")
            print("   F0 RMSE and VUV error metrics may not be reliable.")
            print("   Consider implementing onset alignment (cross-correlation) before F0 extraction.\n")
    
    # Mel-spectrogram statistics (check if values are reasonable)
    if 'mel_real_mean' in df.columns:
        print(f"Mel-Spec Real:  mean={df['mel_real_mean'].mean():.2f} ± {df['mel_real_std'].mean():.2f} dB, " +
              f"range=[{df['mel_real_min'].mean():.1f}, {df['mel_real_max'].mean():.1f}] dB")
    if 'mel_synth_mean' in df.columns:
        print(f"Mel-Spec Synth: mean={df['mel_synth_mean'].mean():.2f} ± {df['mel_synth_std'].mean():.2f} dB, " +
              f"range=[{df['mel_synth_min'].mean():.1f}, {df['mel_synth_max'].mean():.1f}] dB")
    if 'mel_frame_dist_mean' in df.columns:
        print(f"Mel-Spec Frame Distance: {df['mel_frame_dist_mean'].mean():.2f} ± {df['mel_frame_dist_std'].mean():.2f} dB")
        print(f"  (Expected: ~1-5 dB for good TTS, >10 dB indicates major mismatch)")
    
    # Audio length statistics
    if 'audio_real_orig_len' in df.columns:
        real_orig_ms = df['audio_real_orig_len'].mean() / args.sr * 1000
        synth_orig_ms = df['audio_synth_orig_len'].mean() / args.sr * 1000
        print(f"Audio length (orig): real={real_orig_ms:.0f}ms, synth={synth_orig_ms:.0f}ms")
    if 'audio_real_trimmed_len' in df.columns:
        real_trim_ms = df['audio_real_trimmed_len'].mean() / args.sr * 1000
        synth_trim_ms = df['audio_synth_trimmed_len'].mean() / args.sr * 1000
        print(f"Audio length (trimmed): real={real_trim_ms:.0f}ms, synth={synth_trim_ms:.0f}ms")
    
    # Per-speaker statistics
    print('\n' + '='*80)
    print('PER-SPEAKER RESULTS')
    print('='*80)
    
    speaker_stats = df.groupby('speaker_id')[core_metrics].agg(['mean', 'std', 'count'])
    print(speaker_stats.to_string())
    
    # Save results
    df.to_csv(args.output, index=False)
    print(f'\n✓ Results saved to: {args.output}')
    
    # Save summary statistics
    summary_path = args.output.replace('.csv', '_summary.csv')
    summary_df = pd.DataFrame({
        'metric': core_metrics,
        'mean': [df[m].mean() for m in core_metrics],
        'std': [df[m].std() for m in core_metrics],
        'count': [df[m].count() for m in core_metrics]
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
