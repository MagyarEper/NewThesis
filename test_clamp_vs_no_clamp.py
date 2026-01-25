#!/usr/bin/env python3
"""
Test: Generate audio with timesteps=20, NO CLAMPING (like sanity_check.py)
"""

import sys
sys.path.append('Grad-TTS')

import torch
import torchaudio
from model import GradTTS
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse
from speechbrain.inference.vocoders import HIFIGAN

print("="*80)
print("TEST: timesteps=20, NO CLAMPING (sanity_check style)")
print("="*80)

# Load model
checkpoint = torch.load('Grad-TTS/logs/hungarian_dysarthria/grad_500.pt', map_location='cpu', weights_only=False)
n_spks = checkpoint['spk_emb.weight'].shape[0]

model = GradTTS(len(symbols)+1, n_spks, 64, 128, 512, 192, 2, 5, 3, 0.1, 4, 80, 48, 0.05, 20.0, 1000)
model.load_state_dict(checkpoint)
model = model.cuda().eval()
print("✓ Model loaded")

# Load vocoder
vocoder = HIFIGAN.from_hparams(source='speechbrain/tts-hifigan-libritts-16kHz', savedir='tmpdir_vocoder')
print("✓ Vocoder loaded")

# Test sentence
cmu = cmudict.CMUDict('Grad-TTS/resources/cmu_dictionary')
text = 'Kapcsold ki a villanyt'

x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).cuda()[None]
x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
spk = torch.LongTensor([0]).cuda()

print(f"\nGenerating: '{text}'")
print("Parameters: timesteps=20, temperature=1.5, length_scale=0.91")

# Generate with timesteps=20, NO CLAMPING
with torch.no_grad():
    y_enc, y_dec, attn = model.forward(x, x_lengths, n_timesteps=20, temperature=1.5, stoc=False, spk=spk, length_scale=0.91)

print(f"\nMel range (NO CLAMPING): [{y_dec.min():.4f}, {y_dec.max():.4f}]")
print(f"Mean: {y_dec.mean():.4f}, Std: {y_dec.std():.4f}")

# Vocode WITHOUT clamping
wav_no_clamp = vocoder.decode_batch(y_dec)
torchaudio.save('test_NO_CLAMP_timesteps20.wav', wav_no_clamp.squeeze(1).cpu(), 16000)
print("\n✓ Saved: test_NO_CLAMP_timesteps20.wav")

# Now WITH clamping
y_dec_clamped = torch.clamp(y_dec, min=-11.5129, max=-3.0)
print(f"\nMel range (WITH CLAMPING): [{y_dec_clamped.min():.4f}, {y_dec_clamped.max():.4f}]")

wav_clamped = vocoder.decode_batch(y_dec_clamped)
torchaudio.save('test_WITH_CLAMP_timesteps20.wav', wav_clamped.squeeze(1).cpu(), 16000)
print("✓ Saved: test_WITH_CLAMP_timesteps20.wav")

print("\n" + "="*80)
print("Compare these files:")
print("  1. test_NO_CLAMP_timesteps20.wav")
print("  2. test_WITH_CLAMP_timesteps20.wav")
print("  3. sanity_check_output/...wav (from previous run)")
print("="*80)
