#!/usr/bin/env python3
"""
Inference Sanity Check Script

Purpose: Quick sanity check after training to ensure model hasn't collapsed.
Tests 10 fixed Hungarian sentences across multiple speakers.

Usage:
    python sanity_check.py --checkpoint logs/hungarian_dysarthria/grad_500.pt --output-dir sanity_check_output

Output: WAV files in output directory, one per sentence per speaker tested.
"""

import argparse
import os
import datetime as dt
import numpy as np
from scipy.io.wavfile import write

import torch

import params
from model import GradTTS
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse

from speechbrain.inference.vocoders import HIFIGAN


# Fixed test sentences (same every time for consistency)
# Based on actual training data from Hungarian Dysarthria Database
SANITY_CHECK_SENTENCES = [
    "Kapcsold ki a villanyt",
    "Nyisd ki az ablakot",
    "Kapcsold be a rádiót",
    "Csukd be az ajtót",
    "Kapcsold ki a televíziót",
    "Kapcsold be a fűtést",
    "Nyisd ki az ajtót a konyhába",
    "Kapcsold be a ventilátort",
    "Zárjuk be az ablakot",
    "Kapcsoljuk ki a porszívót"
]


def synthesize_sentence(generator, vocoder, text, speaker_id, cmu_dict, timesteps=10, 
                        length_scale=0.91, temperature=1.5, stoc=False):
    """
    Synthesize a single sentence.
    
    Args:
        length_scale: Duration multiplier (>1.0 = longer, <1.0 = shorter)
        temperature: Sampling temperature (lower = more stable, higher = more varied)
        stoc: Stochastic sampling flag
    
    Returns:
        audio (numpy array): Generated audio samples
        rtf (float): Real-time factor (lower is faster)
    """
    # Convert text to phoneme sequence
    x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu_dict), len(symbols))).cuda()[None]
    x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
    
    # Set speaker
    spk = torch.LongTensor([speaker_id]).cuda() if speaker_id is not None else None
    
    # Generate mel-spectrogram
    t_start = dt.datetime.now()
    y_enc, y_dec, attn = generator.forward(
        x, x_lengths, 
        n_timesteps=timesteps, 
        temperature=temperature,
        stoc=stoc, 
        spk=spk, 
        length_scale=length_scale
    )
    generation_time = (dt.datetime.now() - t_start).total_seconds()
    
    # Calculate RTF (Real-Time Factor)
    audio_duration = y_dec.shape[-1] * 256 / 16000  # samples / sample_rate
    rtf = generation_time / audio_duration
    
    # Convert mel to audio using HiFi-GAN
    # y_dec comes from Grad-TTS as [batch, n_mels, time] (confirmed from training test)
    # SpeechBrain HiFi-GAN expects: [batch, n_mels, time] - NO TRANSPOSE NEEDED!
    
    # Ensure 3D tensor [batch, n_mels, time]
    if y_dec.dim() == 2:  # [n_mels, time]
        y_dec = y_dec.unsqueeze(0)  # [1, n_mels, time]
    
    # Vocoder expects [batch, n_mels, time] - already correct!
    audio_tensor = vocoder.decode_batch(y_dec)
    audio = (audio_tensor.cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
    
    return audio, rtf


def main():
    parser = argparse.ArgumentParser(description='Inference sanity check with fixed test sentences')
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to Grad-TTS checkpoint (e.g., logs/hungarian_dysarthria/grad_500.pt)')
    parser.add_argument('--output-dir', type=str, default='sanity_check_output',
                        help='Output directory for generated WAV files')
    parser.add_argument('--timesteps', type=int, default=10,
                        help='Number of diffusion timesteps (default: 10)')
    parser.add_argument('--speakers', type=int, nargs='+', default=[0, 5, 10],
                        help='Speaker IDs to test (default: 0 5 10)')
    parser.add_argument('--length-scale', type=float, default=0.91,
                        help='Duration scale factor (default: 0.91, try 1.0 for longer)')
    parser.add_argument('--temperature', type=float, default=1.5,
                        help='Sampling temperature (default: 1.5, lower = more stable)')
    parser.add_argument('--stoc', action='store_true',
                        help='Use stochastic sampling (default: False)')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print('='*80)
    print('INFERENCE SANITY CHECK')
    print('='*80)
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Output directory: {args.output_dir}')
    print(f'Test sentences: {len(SANITY_CHECK_SENTENCES)}')
    print(f'Speakers to test: {args.speakers}')
    print(f'Parameters: length_scale={args.length_scale}, temperature={args.temperature}, '
          f'timesteps={args.timesteps}, stoc={args.stoc}')
    print('='*80)
    
    # Initialize Grad-TTS
    print('\nInitializing Grad-TTS...')
    generator = GradTTS(
        len(symbols)+1, params.n_spks, params.spk_emb_dim,
        params.n_enc_channels, params.filter_channels,
        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
        params.enc_kernel, params.enc_dropout, params.window_size,
        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale
    )
    generator.load_state_dict(torch.load(args.checkpoint, map_location=lambda loc, storage: loc))
    generator = generator.cuda().eval()
    print(f'Number of parameters: {generator.nparams}')
    
    # Initialize HiFi-GAN vocoder
    print('Initializing HiFi-GAN vocoder...')
    vocoder = HIFIGAN.from_hparams(
        source="speechbrain/tts-hifigan-libritts-16kHz",
        savedir="pretrained_models/tts-hifigan-libritts-16kHz"
    )
    
    # Load CMU dictionary
    cmu = cmudict.CMUDict('./resources/cmu_dictionary')
    
    # Synthesize all sentences
    print('\nStarting synthesis...\n')
    rtf_values = []
    
    with torch.no_grad():
        for spk_id in args.speakers:
            print(f'Speaker {spk_id:02d}:')
            for i, text in enumerate(SANITY_CHECK_SENTENCES):
                try:
                    # Synthesize
                    audio, rtf = synthesize_sentence(
                        generator, vocoder, text, spk_id, cmu, 
                        timesteps=args.timesteps,
                        length_scale=args.length_scale,
                        temperature=args.temperature,
                        stoc=args.stoc
                    )
                    rtf_values.append(rtf)
                    
                    # Save WAV file
                    output_path = os.path.join(args.output_dir, f'spk{spk_id:02d}_sent{i:02d}.wav')
                    write(output_path, 16000, audio)
                    
                    # Print progress
                    print(f'  [{i+1:2d}/{len(SANITY_CHECK_SENTENCES)}] RTF: {rtf:.3f} - "{text[:40]}..."')
                    
                except Exception as e:
                    print(f'  [{i+1:2d}/{len(SANITY_CHECK_SENTENCES)}] ERROR: {str(e)}')
            
            print()
    
    # Summary statistics
    print('='*80)
    print('SANITY CHECK COMPLETE')
    print('='*80)
    print(f'Total files generated: {len([f for f in os.listdir(args.output_dir) if f.endswith(".wav")])}')
    
    if rtf_values:
        print(f'Average RTF: {np.mean(rtf_values):.3f}')
        print(f'Min RTF: {np.min(rtf_values):.3f}')
        print(f'Max RTF: {np.max(rtf_values):.3f}')
    else:
        print('No successful syntheses - check errors above')
    
    print(f'\nOutput directory: {args.output_dir}')
    print('\nNext steps:')
    print('1. Listen to generated samples')
    print('2. Check for audio quality (noise, collapse, intelligibility)')
    print('3. If samples sound good → proceed to full evaluation')
    print('4. If samples are noisy/collapsed → debug model/training')
    print('='*80)


if __name__ == '__main__':
    main()
