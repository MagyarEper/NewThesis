#!/usr/bin/env python3
"""
Test the range and format of generated mel-spectrograms from Grad-TTS
"""

import torch
import numpy as np

# Load a few generated mel-spectrograms
mel_files = [
    'spectrograms/C_001_0001_window_smallroom_AOL.pt',
    'spectrograms/C_001_0010_kettle_NULL_AO.pt',
    'spectrograms/C_001_0100_shutter_kitchen_AOL.pt'
]

print("="*80)
print("GENERATED MEL-SPECTROGRAM ANALYSIS")
print("="*80)

for mel_file in mel_files:
    mel = torch.load(mel_file)
    
    print(f"\nFile: {mel_file}")
    print(f"  Shape: {mel.shape}")
    print(f"  Range: [{mel.min():.4f}, {mel.max():.4f}]")
    print(f"  Mean: {mel.mean():.4f}")
    print(f"  Std: {mel.std():.4f}")

print("\n" + "="*80)
print("EXPECTED RANGE (from test_mel_format.py):")
print("="*80)
print("  WITH compression=True + min_max_norm=True:")
print("    Range: [-11.5129, -3.2701]")
print("    Mean: -9.1219")
print("    Std: 1.5945")

print("\n" + "="*80)
print("DIAGNOSIS:")
print("="*80)
print("If generated mel range is VERY different from expected,")
print("then the vocoder receives wrong format → bad audio quality")
print("="*80)
