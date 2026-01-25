#!/usr/bin/env python3
"""
Test Whisper WER on REAL dysarthric audio to establish baseline.
"""

import whisper
import jiwer
import re

def normalize_text(text):
    """Basic Hungarian text normalization."""
    text = text.lower()
    text = re.sub(r'[^\w\s áéíóöőúüű]', '', text)
    text = ' '.join(text.split())
    return text


print("Loading Whisper base...")
model = whisper.load_model("base")
print("✓ Model loaded\n")

# Test cases from sweep sentences
test_cases = [
    ("wavs_16khz/C_001_0001_stove_NULL_AO.wav", "Kapcsold ki a hűtő ő villanytűzhelyt"),
    ("wavs_16khz/C_001_0010_heating_NULL_AO.wav", "Kapcsold be a rá radiátort"),
    ("wavs_16khz/C_001_0100_shutter_kitchen_AOL.wav", "Ereszd le a zsala zsalut a konyhába"),
    ("wavs_16khz/C_002_0010_window_smallroom_AOL.wav", "Zárjuk be az ablakot a kisszobába"),
    ("wavs_16khz/C_002_0001_vacuum_room_LAO.wav", "Szobába kapcsoljuk ki a porszívót"),
]

print("="*80)
print("WHISPER WER ON REAL DYSARTHRIC AUDIO")
print("="*80)

wer_list = []
cer_list = []

for audio_path, reference in test_cases:
    result = model.transcribe(audio_path, language='hu', task='transcribe')
    transcription = result['text']
    
    # Normalize
    ref_norm = normalize_text(reference)
    trans_norm = normalize_text(transcription)
    
    # Compute WER/CER
    wer = jiwer.wer(ref_norm, trans_norm)
    cer = jiwer.cer(ref_norm, trans_norm)
    
    wer_list.append(wer)
    cer_list.append(cer)
    
    print(f"\nFile: {audio_path.split('/')[-1]}")
    print(f"  Reference:     {reference}")
    print(f"  Transcription: {transcription}")
    print(f"  WER: {wer*100:.1f}%, CER: {cer*100:.1f}%")

print("\n" + "="*80)
print(f"AVERAGE WER ON REAL AUDIO: {sum(wer_list)/len(wer_list)*100:.1f}%")
print(f"AVERAGE CER ON REAL AUDIO: {sum(cer_list)/len(cer_list)*100:.1f}%")
print("="*80)

if sum(wer_list)/len(wer_list) > 0.5:
    print("\n⚠️  WARNING: Whisper struggles with dysarthric speech!")
    print("   High WER on real audio means Whisper is not a reliable metric.")
    print("   Synthetic audio WER comparison is not meaningful.")
else:
    print("\n✓ Whisper understands real dysarthric audio well.")
    print("  Synthetic audio WER can be used as a quality metric.")
