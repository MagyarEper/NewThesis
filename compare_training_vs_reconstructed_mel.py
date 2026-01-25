#!/usr/bin/env python3
"""
Compare: Training mel vs Vocoder reconstructed mel
If they match → vocoder works perfectly
If they don't → something converts the format
"""

import sys
sys.path.append('Grad-TTS')

import torch
import torchaudio
import librosa
import numpy as np
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram

print("="*80)
print("MEL FORMAT COMPARISON TEST")
print("="*80)

# Test file (first sample from training dataset)
test_file = 'wavs_16khz/C_001_0010_window_NULL_AO.wav'
reconstructed_file = 'dataset_mel_reconstructed_wavs/dataset_sample_0_reconstructed.wav'

# 1. Extract training-format mel from ORIGINAL
print("\n1. Original audio → Training mel (compression=True)")
signal, sr = torchaudio.load(test_file)
signal = signal[0]

mel_training, _ = mel_spectogram(
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

print(f"   Training mel shape: {mel_training.shape}")
print(f"   Range: [{mel_training.min():.4f}, {mel_training.max():.4f}]")
print(f"   Mean: {mel_training.mean():.4f}, Std: {mel_training.std():.4f}")

# 2. Extract mel from RECONSTRUCTED (vocoder output)
print("\n2. Reconstructed audio → Mel (same params)")
signal_recon, sr = torchaudio.load(reconstructed_file)
signal_recon = signal_recon[0]

mel_reconstructed, _ = mel_spectogram(
    audio=signal_recon,
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

print(f"   Reconstructed mel shape: {mel_reconstructed.shape}")
print(f"   Range: [{mel_reconstructed.min():.4f}, {mel_reconstructed.max():.4f}]")
print(f"   Mean: {mel_reconstructed.mean():.4f}, Std: {mel_reconstructed.std():.4f}")

# 3. Compute distance
print("\n3. Mel distance (Euclidean)")
# Align lengths
min_len = min(mel_training.shape[1], mel_reconstructed.shape[1])
mel_training_cut = mel_training[:, :min_len]
mel_reconstructed_cut = mel_reconstructed[:, :min_len]

mse = torch.mean((mel_training_cut - mel_reconstructed_cut)**2).item()
mae = torch.mean(torch.abs(mel_training_cut - mel_reconstructed_cut)).item()

print(f"   MSE: {mse:.6f}")
print(f"   MAE: {mae:.6f}")

print("\n" + "="*80)
print("INTERPRETATION:")
print("="*80)
if mae < 0.5:
    print("✓ Vocoder perfectly preserves mel format (MAE < 0.5)")
    print("  → Training mel and vocoder output mel match!")
elif mae < 2.0:
    print("⚠ Vocoder partially preserves mel (0.5 < MAE < 2.0)")
    print("  → Some information loss but format is correct")
else:
    print("✗ Vocoder changes mel format significantly (MAE > 2.0)")
    print("  → Major format mismatch!")

print("="*80)
