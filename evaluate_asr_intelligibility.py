#!/usr/bin/env python3
"""
ASR-based Intelligibility Evaluation for TTS

Computes WER (Word Error Rate) and CER (Character Error Rate) using Whisper ASR
to measure speech intelligibility of generated audio.

This is complementary to STOI metrics - ASR measures actual transcription accuracy.

Usage:
    python evaluate_asr_intelligibility.py \
        --audio-dir generated_test_wavs \
        --manifest test_manifest.csv \
        --output asr_results.csv \
        --model openai/whisper-large-v3
"""

import argparse
import os
import csv
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration
)

# WER/CER metrics
try:
    import evaluate
    EVALUATE_AVAILABLE = True
except ImportError:
    print("Warning: 'evaluate' library not available. Install with: pip install evaluate jiwer")
    EVALUATE_AVAILABLE = False


def normalize_hungarian_text(text):
    """
    Basic text normalization for Hungarian (lowercase, remove punctuation).
    
    Args:
        text: Input text
        
    Returns:
        normalized_text: Normalized text
    """
    import re
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation (keep Hungarian letters: á, é, í, ó, ö, ő, ú, ü, ű)
    text = re.sub(r'[^\w\s áéíóöőúüű]', '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text


def evaluate_asr_pair(audio_path, reference_text, model, processor, device, language="hungarian"):
    """
    Transcribe audio with Whisper and compute WER/CER against reference.
    
    Args:
        audio_path: Path to audio file
        reference_text: Ground truth transcription
        model: Whisper model
        processor: Whisper processor
        device: torch device
        language: Language code ("hungarian" or "english")
        
    Returns:
        transcription: ASR transcription
        wer: Word Error Rate (0-1, lower is better)
        cer: Character Error Rate (0-1, lower is better)
    """
    import librosa
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Prepare input
    input_features = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features.to(device)
    
    # Generate transcription
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    
    # Decode
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    # Normalize both texts
    transcription_norm = normalize_hungarian_text(transcription)
    reference_norm = normalize_hungarian_text(reference_text)
    
    # Compute WER/CER
    if EVALUATE_AVAILABLE:
        metric_wer = evaluate.load("wer")
        metric_cer = evaluate.load("cer")
        
        wer = metric_wer.compute(predictions=[transcription_norm], references=[reference_norm])
        cer = metric_cer.compute(predictions=[transcription_norm], references=[reference_norm])
    else:
        wer = np.nan
        cer = np.nan
    
    return transcription, wer, cer


def main():
    parser = argparse.ArgumentParser(description='ASR-based intelligibility evaluation')
    parser.add_argument('--audio-dir', type=str, required=True,
                        help='Directory with audio files (real or synthetic)')
    parser.add_argument('--manifest', type=str, required=True,
                        help='Manifest file with transcriptions')
    parser.add_argument('--output', type=str, default='asr_results.csv',
                        help='Output CSV file')
    parser.add_argument('--model', type=str, default='openai/whisper-large-v3',
                        help='Whisper model (default: openai/whisper-large-v3)')
    parser.add_argument('--language', type=str, default='hungarian',
                        choices=['hungarian', 'english'],
                        help='Language for ASR (default: hungarian)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda/cpu)')
    args = parser.parse_args()
    
    if not EVALUATE_AVAILABLE:
        print("ERROR: 'evaluate' library required. Install with:")
        print("  pip install evaluate jiwer")
        return
    
    print('='*80)
    print('ASR-BASED INTELLIGIBILITY EVALUATION')
    print('='*80)
    print(f'Audio directory: {args.audio_dir}')
    print(f'Manifest: {args.manifest}')
    print(f'Whisper model: {args.model}')
    print(f'Language: {args.language}')
    print(f'Device: {args.device}')
    print('='*80)
    
    # Load Whisper model
    print('\nLoading Whisper model...')
    device = torch.device(args.device)
    
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model = model.to(device)
    model.eval()
    
    # Set language
    if args.language == "hungarian":
        model.generation_config.language = "hungarian"
        model.generation_config.task = "transcribe"
    else:
        model.generation_config.language = "english"
        model.generation_config.task = "transcribe"
    
    print('✓ Model loaded')
    
    # Load manifest
    print('\nLoading manifest...')
    
    if args.manifest.endswith('.csv'):
        with open(args.manifest, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            lines = [(row['wav'], row['text'], row['speaker']) for row in reader]
    else:
        # Pipe-separated format: path|text|speaker
        with open(args.manifest, 'r', encoding='utf-8') as f:
            lines = [line.strip().split('|') for line in f.readlines()]
    
    print(f'Found {len(lines)} utterances')
    
    # Evaluate
    print('\nTranscribing and evaluating...')
    results = []
    
    wer_list = []
    cer_list = []
    
    with open(args.output, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'utt_id',
            'speaker_id',
            'reference_text',
            'transcription',
            'wer',
            'cer',
            'word_count',
            'char_count'
        ])
        
        for wav_path, text, speaker_id in tqdm(lines):
            basename = Path(wav_path).name
            audio_path = os.path.join(args.audio_dir, basename)
            
            if not os.path.exists(audio_path):
                print(f"Warning: File not found: {audio_path}")
                continue
            
            # Transcribe and evaluate
            try:
                transcription, wer, cer = evaluate_asr_pair(
                    audio_path, text, model, processor, device, args.language
                )
                
                # Count words/chars
                ref_norm = normalize_hungarian_text(text)
                word_count = len(ref_norm.split())
                char_count = len(ref_norm.replace(' ', ''))
                
                wer_list.append(wer)
                cer_list.append(cer)
                
                writer.writerow([
                    basename,
                    speaker_id,
                    text,
                    transcription,
                    f"{wer:.4f}" if not np.isnan(wer) else "NaN",
                    f"{cer:.4f}" if not np.isnan(cer) else "NaN",
                    word_count,
                    char_count
                ])
                
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                continue
    
    # Compute overall statistics
    print('\n' + '='*80)
    print('OVERALL RESULTS')
    print('='*80)
    
    wer_array = np.array(wer_list)
    cer_array = np.array(cer_list)
    
    print(f'WER (Word Error Rate): {np.mean(wer_array):.3f} ± {np.std(wer_array):.3f}')
    print(f'CER (Character Error Rate): {np.mean(cer_array):.3f} ± {np.std(cer_array):.3f}')
    print(f'Samples evaluated: {len(wer_list)}')
    
    print('\n' + '='*80)
    print('INTERPRETATION:')
    print('='*80)
    
    wer_mean = np.mean(wer_array)
    
    if wer_mean < 0.10:
        print('✓ Excellent intelligibility (WER < 10%)')
    elif wer_mean < 0.20:
        print('✓ Good intelligibility (WER 10-20%)')
    elif wer_mean < 0.30:
        print('⚠️  Moderate intelligibility (WER 20-30%)')
    elif wer_mean < 0.50:
        print('⚠️  Poor intelligibility (WER 30-50%)')
    else:
        print('❌ Very poor intelligibility (WER > 50%)')
    
    print(f'\n✓ Results saved to: {args.output}')
    print('='*80)


if __name__ == '__main__':
    main()
