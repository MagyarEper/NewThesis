#!/usr/bin/env python3
"""
Generate synthetic audio from a Grad-TTS checkpoint for any manifest split.

Works with both CSV manifests (utt_id,wav,speaker,text) and
pipe-separated manifests (path|text|speaker_id).

Use --output-manifest to also produce a CSV manifest for the generated files
(needed for Whisper fine-tuning on synthetic data).

Examples:
    # Test set (original usage)
    python generate_test_set.py \
        --checkpoint Grad-TTS/logs/hungarian_dysarthria/grad_500.pt \
        --manifest test_manifest.csv \
        --output-dir generated_test_wavs

    # Train set with output manifest for Whisper fine-tuning
    python generate_test_set.py \
        --checkpoint Grad-TTS/logs/hungarian_dysarthria/grad_500.pt \
        --manifest train_manifest.csv \
        --output-dir synthetic_train_wavs \
        --output-manifest synthetic_train_manifest.csv \
        --timesteps 10
"""

import argparse
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
import soundfile as sf

# Import model components
sys.path.append('Grad-TTS')
from model import GradTTS
from text import text_to_sequence
from text.symbols import symbols
from utils import intersperse

# Import vocoder (will be loaded dynamically to handle API changes)
# from speechbrain.pretrained import HIFIGAN  # Old API
# from speechbrain.inference.vocoders import HIFIGAN  # New API


def load_model(checkpoint_path, n_spks=39):
    """Load trained GradTTS model."""
    print(f'Loading checkpoint: {checkpoint_path}')
    
    # Load checkpoint first to check vocab size
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        epoch = checkpoint.get('epoch', 'unknown')
    else:
        state_dict = checkpoint
        epoch = 'unknown'
    
    # Get actual vocab size from checkpoint
    if 'encoder.emb.weight' in state_dict:
        actual_n_vocab = state_dict['encoder.emb.weight'].shape[0]
        print(f'Detected vocab size from checkpoint: {actual_n_vocab}')
    else:
        actual_n_vocab = len(symbols)
        print(f'Using default vocab size: {actual_n_vocab}')
    
    # Model parameters (MUST MATCH training params.py)
    spk_emb_dim = 64
    n_enc_channels = 128  # From params.py (optimized for 16GB VRAM)
    filter_channels = 512  # From params.py
    filter_channels_dp = 192  # From params.py
    n_enc_layers = 5  # From params.py (was reduced from 6)
    enc_kernel = 3
    enc_dropout = 0.1
    n_heads = 2
    window_size = 4
    
    n_feats = 80
    dec_dim = 48  # From params.py (was reduced from 64)
    beta_min = 0.05
    beta_max = 20.0
    pe_scale = 1000
    
    # Initialize model with actual vocab size from checkpoint
    model = GradTTS(
        actual_n_vocab,  # Use vocab size from checkpoint
        n_spks,
        spk_emb_dim,
        n_enc_channels,
        filter_channels,
        filter_channels_dp, 
        n_heads, 
        n_enc_layers,
        enc_kernel, 
        enc_dropout, 
        window_size, 
        n_feats, 
        dec_dim, 
        beta_min, 
        beta_max, 
        pe_scale
    )
    
    # Load state dict
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f'✓ Model loaded from epoch {epoch}')
    
    if torch.cuda.is_available():
        model = model.cuda()
        print('✓ Model moved to GPU')
    
    return model


def load_vocoder():
    """Load HiFi-GAN vocoder."""
    print('Loading HiFi-GAN vocoder...')
    
    # Lazy import to avoid CUDA errors on broken torchaudio installations
    try:
        # Try new SpeechBrain API
        from speechbrain.inference.vocoders import HIFIGAN
        print('Using speechbrain.inference.vocoders.HIFIGAN (new API)')
        vocoder = HIFIGAN.from_hparams(
            source="speechbrain/tts-hifigan-libritts-16kHz",
            savedir="tmpdir_vocoder"
        )
        print('✓ Vocoder loaded')
        return vocoder
    except (ImportError, OSError) as e:
        print(f'New API failed: {e}')
        pass
    
    try:
        # Try old SpeechBrain API
        from speechbrain.pretrained import HIFIGAN
        print('Using speechbrain.pretrained.HIFIGAN (old API)')
        vocoder = HIFIGAN.from_hparams(
            source="speechbrain/tts-hifigan-libritts-16kHz",
            savedir="tmpdir_vocoder"
        )
        print('✓ Vocoder loaded')
        return vocoder
    except (ImportError, OSError) as e:
        print(f'Old API also failed: {e}')
        raise RuntimeError("Could not load HiFi-GAN vocoder. Please check your PyTorch/torchaudio installation.")
    


def synthesize_utterance(model, vocoder, text, speaker_id, 
                         length_scale=1.0, temperature=1.2, timesteps=20, stoc=False):
    """
    Synthesize one utterance.
    
    Args:
        model: GradTTS model
        vocoder: HiFi-GAN vocoder
        text: Text to synthesize
        speaker_id: Speaker ID (integer)
        length_scale: Length scale (speed control)
        temperature: Sampling temperature
        timesteps: Number of diffusion steps
        stoc: Use stochastic sampling
        
    Returns:
        wav: Waveform (numpy array)
        rtf: Real-time factor
    """
    # Prepare text
    x = torch.LongTensor(intersperse(text_to_sequence(text, cleaner_names=['basic_cleaners'], dictionary=None), len(symbols))).unsqueeze(0)
    x_lengths = torch.LongTensor([x.shape[-1]])
    
    # Speaker ID
    spk = torch.LongTensor([speaker_id])
    
    if torch.cuda.is_available():
        x = x.cuda()
        x_lengths = x_lengths.cuda()
        spk = spk.cuda()
    
    # Generate mel-spectrogram
    start_time = time.time()
    
    with torch.no_grad():
        y_enc, y_dec, attn = model.forward(
            x, 
            x_lengths,
            n_timesteps=timesteps, 
            temperature=temperature,
            stoc=stoc, 
            spk=spk,
            length_scale=length_scale
        )
    
    # y_dec shape: [batch, n_mels, time] — correct format for decode_batch

    # Generate waveform with vocoder
    waveforms = vocoder.decode_batch(y_dec)
    wav = waveforms.squeeze().cpu().numpy()
    
    # Compute RTF
    generation_time = time.time() - start_time
    audio_duration = len(wav) / 16000  # 16kHz sample rate
    rtf = generation_time / audio_duration
    
    return wav, rtf


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic audio from Grad-TTS')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--manifest', type=str, required=True,
                        help='Manifest file (CSV or pipe-separated)')
    parser.add_argument('--output-dir', type=str, default='generated_test_wavs',
                        help='Output directory for generated WAVs')
    parser.add_argument('--output-manifest', type=str, default=None,
                        help='If set, write a CSV manifest for generated files (utt_id,wav,speaker,text)')
    parser.add_argument('--length-scale', type=float, default=1.0,
                        help='Length scale (speed control, default: 1.0)')
    parser.add_argument('--temperature', type=float, default=1.2,
                        help='Sampling temperature (default: 1.2)')
    parser.add_argument('--timesteps', type=int, default=20,
                        help='Number of diffusion steps (default: 20)')
    parser.add_argument('--stoc', action='store_true',
                        help='Use stochastic sampling')
    parser.add_argument('--n-spks', type=int, default=39,
                        help='Number of speakers in model (default: 39)')
    args = parser.parse_args()
    
    print('='*80)
    print('GENERATING SYNTHETIC AUDIO')
    print('='*80)
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Manifest: {args.manifest}')
    print(f'Output directory: {args.output_dir}')
    if args.output_manifest:
        print(f'Output manifest: {args.output_manifest}')
    print(f'Parameters:')
    print(f'  length_scale = {args.length_scale}')
    print(f'  temperature  = {args.temperature}')
    print(f'  timesteps    = {args.timesteps}')
    print(f'  stochastic   = {args.stoc}')
    print('='*80)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and vocoder
    model = load_model(args.checkpoint, n_spks=args.n_spks)
    vocoder = load_vocoder()
    
    # Load manifest
    # lines: list of (wav_path, text, speaker_id_int, speaker_name_str_or_None)
    print(f'\nLoading manifest: {args.manifest}')
    
    if args.manifest.endswith('.csv'):
        # CSV format with headers: utt_id,wav,speaker,text
        import csv
        with open(args.manifest, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # Create speaker ID mapping (C_001 -> 0, C_002 -> 1, etc.)
        unique_speakers = sorted(set(row['speaker'] for row in rows))
        speaker_to_id = {spk: idx for idx, spk in enumerate(unique_speakers)}
        print(f'Found {len(unique_speakers)} unique speakers: {unique_speakers[:5]}...')
        
        # Keep speaker names for output manifest
        lines = [(row['wav'], row['text'], speaker_to_id[row['speaker']],
                  row.get('utt_id', Path(row['wav']).stem), row['speaker'])
                 for row in rows]
    else:
        # Pipe-separated format: path|text|speaker_id (already integers)
        with open(args.manifest, 'r', encoding='utf-8') as f:
            raw = [line.strip().split('|') for line in f.readlines()]
        lines = [(wav, text, int(spk), Path(wav).stem, None)
                 for wav, text, spk in raw]
    
    print(f'Found {len(lines)} utterances')
    
    # Generate all utterances
    print('\nGenerating...')
    rtf_values = []
    failed = []
    output_rows = []  # for output manifest
    
    for wav_path, text, speaker_id, utt_id, speaker_name in tqdm(lines):
        try:
            # Get output filename — use utt_id if wav_path is empty
            basename = Path(wav_path).name if wav_path else f'{utt_id}.wav'
            output_path = os.path.join(args.output_dir, basename)
            
            # Skip if already exists
            if os.path.exists(output_path):
                if speaker_name is not None:
                    output_rows.append({
                        'utt_id': utt_id,
                        'wav': os.path.abspath(output_path),
                        'speaker': speaker_name,
                        'text': text,
                    })
                continue
            
            # Synthesize
            wav, rtf = synthesize_utterance(
                model, vocoder, text, speaker_id,
                length_scale=args.length_scale,
                temperature=args.temperature,
                timesteps=args.timesteps,
                stoc=args.stoc
            )
            
            # Save
            sf.write(output_path, wav, 16000)
            rtf_values.append(rtf)
            
            if speaker_name is not None:
                output_rows.append({
                    'utt_id': utt_id,
                    'wav': os.path.abspath(output_path),
                    'speaker': speaker_name,
                    'text': text,
                })
            
        except Exception as e:
            print(f'\nFailed: {utt_id} - {e}')
            failed.append((utt_id, str(e)))
    
    # Write output manifest if requested
    if args.output_manifest and output_rows:
        import csv
        with open(args.output_manifest, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['utt_id', 'wav', 'speaker', 'text'])
            writer.writeheader()
            writer.writerows(output_rows)
        print(f'\n✓ Output manifest: {args.output_manifest} ({len(output_rows)} rows)')
    
    # Summary
    print('\n' + '='*80)
    print('GENERATION COMPLETE')
    print('='*80)
    print(f'Total utterances: {len(lines)}')
    print(f'Successfully generated: {len(rtf_values)}')
    print(f'Skipped (already exist): {len(lines) - len(rtf_values) - len(failed)}')
    print(f'Failed: {len(failed)}')
    
    if rtf_values:
        print(f'\nAverage RTF: {np.mean(rtf_values):.3f}')
    
    if failed:
        print('\nFailed utterances:')
        for name, error in failed[:10]:
            print(f'  {name}: {error}')
        if len(failed) > 10:
            print(f'  ... and {len(failed) - 10} more')
    
    print(f'\n✓ Output directory: {args.output_dir}')
    print('='*80)


if __name__ == '__main__':
    main()
