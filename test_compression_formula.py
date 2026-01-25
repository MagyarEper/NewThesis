#!/usr/bin/env python3
"""
Reverse engineer what compression=True does in SpeechBrain mel_spectogram
"""

import torch
import torchaudio
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram

# Load audio
signal, sr = torchaudio.load("wavs_16khz/C_001_0001_window_smallroom_AOL.wav")
signal = signal[0]

print("="*80)
print("REVERSE ENGINEERING compression=True")
print("="*80)

# Test 1: compression=True
mel_comp, _ = mel_spectogram(
    audio=signal,
    sample_rate=16000,
    hop_length=256,
    win_length=1024,
    n_mels=80,
    n_fft=1024,
    f_min=0.0,
    f_max=8000.0,
    power=1,  # magnitude
    normalized=False,
    min_max_energy_norm=True,
    norm="slaney",
    mel_scale="slaney",
    compression=True
)

# Test 2: compression=False, then log
mel_no_comp, _ = mel_spectogram(
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

mel_manual_log = torch.log(mel_no_comp + 1e-5)

print("\n1. WITH compression=True:")
print(f"   Range: [{mel_comp.min():.4f}, {mel_comp.max():.4f}]")
print(f"   Mean: {mel_comp.mean():.4f}, Std: {mel_comp.std():.4f}")

print("\n2. WITHOUT compression (raw magnitude, min-max normalized):")
print(f"   Range: [{mel_no_comp.min():.4f}, {mel_no_comp.max():.4f}]")
print(f"   Mean: {mel_no_comp.mean():.4f}, Std: {mel_no_comp.std():.4f}")

print("\n3. Manual log(mel_no_comp + 1e-5):")
print(f"   Range: [{mel_manual_log.min():.4f}, {mel_manual_log.max():.4f}]")
print(f"   Mean: {mel_manual_log.mean():.4f}, Std: {mel_manual_log.std():.4f}")

print("\n" + "="*80)
print("TEST: Is compression=True equivalent to log()?")
print("="*80)

if torch.allclose(mel_comp, mel_manual_log, atol=1e-3):
    print("✓ YES: compression=True == log(mel + 1e-5)")
else:
    print("✗ NO: compression=True is NOT just log()")
    print("\nLet's test other hypotheses...")
    
    # Test sqrt
    mel_sqrt = torch.sqrt(mel_no_comp)
    if torch.allclose(mel_comp, mel_sqrt, atol=1e-3):
        print("✓ compression=True == sqrt()")
    else:
        print("✗ compression is not sqrt() either")
    
    # Test cube root
    mel_cbrt = torch.pow(mel_no_comp, 1/3)
    if torch.allclose(mel_comp, mel_cbrt, atol=1e-3):
        print("✓ compression=True == cbrt()")
    else:
        print("✗ compression is not cbrt()")
    
    # Check ratio
    print("\n Trying to find compression formula...")
    print(f"   mel_comp / mel_no_comp: {(mel_comp / (mel_no_comp + 1e-10)).mean():.4f}")
    print(f"   mel_comp - mel_no_comp: {(mel_comp - mel_no_comp).mean():.4f}")

print("\n" + "="*80)
