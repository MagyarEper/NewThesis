#!/usr/bin/env python3
"""
Test what SpeechBrain compression does to mel-spectrograms
"""

import torch
import torchaudio
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram

# Load a sample audio
signal, sr = torchaudio.load("wavs_16khz/C_001_0001_window_smallroom_AOL.wav")
signal = signal[0]  # mono

print("="*80)
print("TESTING MEL-SPECTROGRAM FORMATS")
print("="*80)

# Test 1: With compression=True (current training/vocoder)
mel_compressed, _ = mel_spectogram(
    audio=signal,
    sample_rate=16000,
    hop_length=256,
    win_length=1024,
    n_mels=80,
    n_fft=1024,
    f_min=0.0,
    f_max=8000.0,
    power=1,
    normalized=False,
    min_max_energy_norm=True,
    norm="slaney",
    mel_scale="slaney",
    compression=True
)

# Test 2: Without compression (standard mel)
mel_uncompressed, _ = mel_spectogram(
    audio=signal,
    sample_rate=16000,
    hop_length=256,
    win_length=1024,
    n_mels=80,
    n_fft=1024,
    f_min=0.0,
    f_max=8000.0,
    power=1,
    normalized=False,
    min_max_energy_norm=True,
    norm="slaney",
    mel_scale="slaney",
    compression=False
)

# Test 3: Standard log-mel (no compression, no min-max norm)
mel_standard, _ = mel_spectogram(
    audio=signal,
    sample_rate=16000,
    hop_length=256,
    win_length=1024,
    n_mels=80,
    n_fft=1024,
    f_min=0.0,
    f_max=8000.0,
    power=2,  # power spectrum
    normalized=False,
    min_max_energy_norm=False,
    norm="slaney",
    mel_scale="slaney",
    compression=False
)

mel_standard_log = torch.log(mel_standard + 1e-5)

print("\n1. WITH COMPRESSION + MIN_MAX_NORM (current):")
print(f"   Shape: {mel_compressed.shape}")
print(f"   Range: [{mel_compressed.min():.4f}, {mel_compressed.max():.4f}]")
print(f"   Mean: {mel_compressed.mean():.4f}")
print(f"   Std: {mel_compressed.std():.4f}")

print("\n2. WITHOUT COMPRESSION (but with min_max_norm):")
print(f"   Shape: {mel_uncompressed.shape}")
print(f"   Range: [{mel_uncompressed.min():.4f}, {mel_uncompressed.max():.4f}]")
print(f"   Mean: {mel_uncompressed.mean():.4f}")
print(f"   Std: {mel_uncompressed.std():.4f}")

print("\n3. STANDARD LOG-MEL (power=2, no compression, no min_max):")
print(f"   Shape: {mel_standard_log.shape}")
print(f"   Range: [{mel_standard_log.min():.4f}, {mel_standard_log.max():.4f}]")
print(f"   Mean: {mel_standard_log.mean():.4f}")
print(f"   Std: {mel_standard_log.std():.4f}")

print("\n" + "="*80)
print("ANALYSIS:")
print("="*80)

# Check if compression is just log
if torch.allclose(mel_compressed, torch.log(mel_uncompressed + 1e-5), atol=1e-3):
    print("✓ compression=True is equivalent to log()")
else:
    print("✗ compression=True is NOT just log()")

print("\n" + "="*80)
