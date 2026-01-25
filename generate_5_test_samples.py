#!/usr/bin/env python3
"""
Generate 5 test audio files with Grad-TTS and analyze mel range
"""

import sys
sys.path.append('Grad-TTS')

import torch
import torchaudio
import numpy as np
import librosa
from model import GradTTS
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse

print("="*80)
print("GRAD-TTS MEL GENERATION TEST")
print("="*80)

# Load model
print("\n1. Loading Grad-TTS model...")
checkpoint_path = 'Grad-TTS/logs/hungarian_dysarthria/grad_500.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# Detect n_spks from checkpoint
if 'spk_emb.weight' in checkpoint:
    n_spks = checkpoint['spk_emb.weight'].shape[0]
    print(f"   Detected n_spks from checkpoint: {n_spks}")
else:
    n_spks = 39  # Default

# Model parameters (MUST MATCH training params.py)
n_enc_channels = 128
filter_channels = 512
filter_channels_dp = 192
n_enc_layers = 5
enc_kernel = 3
enc_dropout = 0.1
n_heads = 2
window_size = 4
n_feats = 80
dec_dim = 48  # CRITICAL: was reduced from 64 in params.py
beta_min = 0.05
beta_max = 20.0
pe_scale = 1000

model = GradTTS(
    len(symbols)+1, n_spks, 64,
    n_enc_channels, filter_channels, filter_channels_dp,
    n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size,
    n_feats, dec_dim, beta_min, beta_max, pe_scale
)

# Load checkpoint (it's directly the state dict)
model.load_state_dict(checkpoint)
model = model.cuda().eval()
print("✓ Model loaded")

# Load vocoder
print("\n2. Loading HiFi-GAN vocoder...")
try:
    from speechbrain.inference.vocoders import HIFIGAN
    vocoder = HIFIGAN.from_hparams(
        source="speechbrain/tts-hifigan-libritts-16kHz",
        savedir="tmpdir_vocoder"
    )
    print("✓ Vocoder loaded")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

# Test texts (HUNGARIAN - from test manifest)
test_texts = [
    "Kapcsold ki a hűtő ő villanytűzhelyt",
    "Kapcsold be a rá radiátort",
    "Ereszd le a zsala zsalut a konyhába",
    "Zárjuk be az ablakot a kisszobába",
    "Kapcsoljuk be a ventilátorokat a kisszobába"
]

cmu = cmudict.CMUDict('Grad-TTS/resources/cmu_dictionary')

print("\n" + "="*80)
print("GENERATING 5 TEST SAMPLES")
print("="*80)

for i, text in enumerate(test_texts):
    print(f"\n{i+1}. Text: '{text}'")
    
    # Text to phonemes
    x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).cuda()[None]
    x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
    spk = torch.LongTensor([0]).cuda()  # Speaker 0
    
    # Generate mel
    with torch.no_grad():
        y_enc, y_dec, attn = model.forward(
            x, x_lengths, 
            n_timesteps=10, 
            temperature=1.2,
            stoc=False, 
            spk=spk, 
            length_scale=1.0
        )
    
    # y_dec shape: [batch, n_mels, time]
    print(f"   Generated mel shape: {y_dec.shape}")
    print(f"   RAW mel range: [{y_dec.min():.4f}, {y_dec.max():.4f}]")
    print(f"   Mean: {y_dec.mean():.4f}, Std: {y_dec.std():.4f}")
    
    # Clamp mel (current fix)
    y_dec_clamped = torch.clamp(y_dec, min=-11.5129, max=-3.0)
    print(f"   CLAMPED mel range: [{y_dec_clamped.min():.4f}, {y_dec_clamped.max():.4f}]")
    
    # Generate audio WITH clamping
    with torch.no_grad():
        wav = vocoder.decode_batch(y_dec_clamped)
    
    # Save
    output_file = f"test_gen_{i+1}_clamped.wav"
    torchaudio.save(output_file, wav.squeeze(1).cpu(), 16000)
    print(f"   → Saved: {output_file}")
    
    # Compute mel from generated audio (for verification)
    wav_np = wav.squeeze().cpu().numpy()
    mel_from_audio = librosa.feature.melspectrogram(
        y=wav_np,
        sr=16000,
        n_fft=1024,
        hop_length=256,
        n_mels=80,
        fmin=0,
        fmax=8000
    )
    mel_from_audio_db = librosa.power_to_db(mel_from_audio, ref=np.max)
    print(f"   Mel from audio (librosa): [{mel_from_audio_db.min():.2f}, {mel_from_audio_db.max():.2f}] dB")

print("\n" + "="*80)
print("EXPECTED TRAINING MEL RANGE:")
print("  compression=True + min_max_norm: [-11.5, -3.3]")
print("\nIF RAW MEL GOES OUTSIDE THIS RANGE:")
print("  → Clamping helps constrain to training range")
print("  → But if audio still sounds bad, vocoder format mismatch!")
print("="*80)
