#!/usr/bin/env python3
"""
Sweep generation parameters to find optimal settings for quality.

Tests combinations of:
- timesteps: 10, 20, 50, 100, 200
- temperature: 0.7, 1.0, 1.3

Evaluates with Whisper WER/CER on 10 fixed Hungarian sentences.
"""

import sys
sys.path.append('Grad-TTS')

import os
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np

from model import GradTTS
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse
from speechbrain.inference.vocoders import HIFIGAN

# Whisper for evaluation
import whisper
import jiwer

# Fixed test sentences (same as sanity check)
TEST_SENTENCES = [
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

# Parameter sweep
TIMESTEPS_VALUES = [10, 20, 50, 100, 200]
TEMPERATURE_VALUES = [0.7, 1.0, 1.3]


def normalize_text(text):
    """Basic Hungarian text normalization."""
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s áéíóöőúüű]', '', text)
    text = ' '.join(text.split())
    return text


def load_models():
    """Load Grad-TTS, vocoder, and Whisper."""
    print("Loading models...")
    
    # Grad-TTS
    checkpoint = torch.load('Grad-TTS/logs/hungarian_dysarthria/grad_500.pt', 
                           map_location='cpu', weights_only=False)
    n_spks = checkpoint['spk_emb.weight'].shape[0]
    
    model = GradTTS(len(symbols)+1, n_spks, 64, 128, 512, 192, 2, 5, 3, 0.1, 4, 
                   80, 48, 0.05, 20.0, 1000)
    model.load_state_dict(checkpoint)
    model = model.cuda().eval()
    print("  ✓ Grad-TTS loaded")
    
    # Vocoder
    vocoder = HIFIGAN.from_hparams(
        source='speechbrain/tts-hifigan-libritts-16kHz',
        savedir='tmpdir_vocoder'
    )
    print("  ✓ HiFi-GAN loaded")
    
    # Whisper
    whisper_model = whisper.load_model("base")
    print("  ✓ Whisper base loaded")
    
    return model, vocoder, whisper_model


def synthesize(model, vocoder, text, cmu, speaker_id=0, 
               timesteps=10, temperature=1.0, length_scale=0.91):
    """Synthesize one utterance."""
    x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), 
                                     len(symbols))).cuda()[None]
    x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
    spk = torch.LongTensor([speaker_id]).cuda()
    
    with torch.no_grad():
        y_enc, y_dec, attn = model.forward(
            x, x_lengths,
            n_timesteps=timesteps,
            temperature=temperature,
            stoc=False,
            spk=spk,
            length_scale=length_scale
        )
    
    # Vocode
    wav = vocoder.decode_batch(y_dec)
    return wav.squeeze().cpu()


def evaluate_with_whisper(whisper_model, audio_path, reference_text):
    """Transcribe with Whisper and compute WER/CER."""
    result = whisper_model.transcribe(
        audio_path,
        language='hu',
        task='transcribe'
    )
    
    transcription = result['text']
    avg_logprob = result.get('avg_logprob', np.nan)
    no_speech_prob = result.get('no_speech_prob', np.nan)
    
    # Normalize
    transcription_norm = normalize_text(transcription)
    reference_norm = normalize_text(reference_text)
    
    # Compute WER/CER
    wer = jiwer.wer(reference_norm, transcription_norm)
    cer = jiwer.cer(reference_norm, transcription_norm)
    
    return {
        'transcription': transcription,
        'wer': wer,
        'cer': cer,
        'avg_logprob': avg_logprob,
        'no_speech_prob': no_speech_prob
    }


def main():
    print("="*80)
    print("GENERATION PARAMETER SWEEP")
    print("="*80)
    print(f"Test sentences: {len(TEST_SENTENCES)}")
    print(f"Timesteps values: {TIMESTEPS_VALUES}")
    print(f"Temperature values: {TEMPERATURE_VALUES}")
    print(f"Total configurations: {len(TIMESTEPS_VALUES) * len(TEMPERATURE_VALUES)}")
    print("="*80)
    
    # Load models
    grad_tts, vocoder, whisper_model = load_models()
    cmu = cmudict.CMUDict('Grad-TTS/resources/cmu_dictionary')
    
    # Output directory
    output_dir = Path('sweep_outputs')
    output_dir.mkdir(exist_ok=True)
    
    results = []
    
    # Sweep
    total_configs = len(TIMESTEPS_VALUES) * len(TEMPERATURE_VALUES)
    with tqdm(total=total_configs * len(TEST_SENTENCES), 
              desc="Sweeping parameters") as pbar:
        
        for timesteps in TIMESTEPS_VALUES:
            for temperature in TEMPERATURE_VALUES:
                config_name = f"t{timesteps}_temp{temperature}"
                config_dir = output_dir / config_name
                config_dir.mkdir(exist_ok=True)
                
                config_results = []
                
                for sent_idx, sentence in enumerate(TEST_SENTENCES):
                    # Generate
                    audio = synthesize(
                        grad_tts, vocoder, sentence, cmu,
                        timesteps=timesteps,
                        temperature=temperature
                    )
                    
                    # Save
                    audio_path = config_dir / f"sent_{sent_idx:02d}.wav"
                    torchaudio.save(str(audio_path), audio.unsqueeze(0), 16000)
                    
                    # Evaluate
                    eval_result = evaluate_with_whisper(
                        whisper_model, str(audio_path), sentence
                    )
                    
                    config_results.append({
                        'timesteps': timesteps,
                        'temperature': temperature,
                        'sentence_idx': sent_idx,
                        'sentence': sentence,
                        'transcription': eval_result['transcription'],
                        'wer': eval_result['wer'],
                        'cer': eval_result['cer'],
                        'avg_logprob': eval_result['avg_logprob'],
                        'no_speech_prob': eval_result['no_speech_prob']
                    })
                    
                    pbar.update(1)
                
                results.extend(config_results)
    
    # Save detailed results
    df = pd.DataFrame(results)
    df.to_csv('sweep_results_detailed.csv', index=False)
    print(f"\n✓ Detailed results saved to: sweep_results_detailed.csv")
    
    # Aggregate by configuration
    summary = df.groupby(['timesteps', 'temperature']).agg({
        'wer': ['mean', 'std'],
        'cer': ['mean', 'std'],
        'avg_logprob': 'mean',
        'no_speech_prob': 'mean'
    }).reset_index()
    
    summary.columns = ['timesteps', 'temperature', 'wer_mean', 'wer_std', 
                      'cer_mean', 'cer_std', 'avg_logprob', 'no_speech_prob']
    summary = summary.sort_values('wer_mean')
    
    summary.to_csv('sweep_results_summary.csv', index=False)
    print(f"✓ Summary saved to: sweep_results_summary.csv")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY (sorted by WER)")
    print("="*80)
    print(summary.to_string(index=False))
    print("="*80)
    
    # Best configuration
    best = summary.iloc[0]
    print(f"\n🏆 BEST CONFIGURATION:")
    print(f"   timesteps={int(best['timesteps'])}, temperature={best['temperature']:.1f}")
    print(f"   WER: {best['wer_mean']*100:.1f}% ± {best['wer_std']*100:.1f}%")
    print(f"   CER: {best['cer_mean']*100:.1f}% ± {best['cer_std']*100:.1f}%")
    print(f"   Avg logprob: {best['avg_logprob']:.3f}")
    print(f"   No speech prob: {best['no_speech_prob']:.3f}")
    print("="*80)


if __name__ == '__main__':
    main()
