#!/usr/bin/env python3
"""
WAV fájlok resample-lése 16 kHz-re a manifest.txt alapján.

Használat:
    python wav_converter.py --manifest manifest.txt --input-dir original_wavs/ --output-dir wavs_16khz/
"""

from pathlib import Path
import librosa
import soundfile as sf
from tqdm import tqdm
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='WAV fájlok resample-lése 16 kHz-re')
    parser.add_argument('--manifest', type=str, default="manifest.txt",
                        help='A manifest fájl (default: manifest.txt)')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Az eredeti WAV fájlok könyvtára')
    parser.add_argument('--output-dir', type=str, default="wavs_16khz",
                        help='Az output könyvtár (default: wavs_16khz)')
    args = parser.parse_args()
    
    manifest_path = Path(args.manifest)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not manifest_path.exists():
        print(f"ERROR: A manifest fájl nem található: {manifest_path}", file=sys.stderr)
        sys.exit(1)
    
    if not input_dir.exists():
        print(f"ERROR: Az input könyvtár nem található: {input_dir}", file=sys.stderr)
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Manifest beolvasása: {manifest_path}")
    
    # Beolvasás pipe-separated formátumból
    data = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) == 3:
                # A wav path-ből kinyerjük a fájlnevet
                wav_path = Path(parts[0])
                filename = wav_path.name
                data.append({
                    'original': input_dir / filename,
                    'output': output_dir / filename
                })

    print(f"Resample-lés 16 kHz-re...")
    print(f"   Input: {input_dir}")
    print(f"   Output: {output_dir}")
    print(f"   Összesen {len(data)} fájl feldolgozása\n")

    success = 0
    failed = 0
    skipped = 0
    
    for item in tqdm(data, desc="Converting"):
        try:
            # Ha az output már létezik, skip
            if item['output'].exists():
                skipped += 1
                continue
            
            # Ha az input nem létezik, skip
            if not item['original'].exists():
                failed += 1
                continue
            
            # Eredeti wav betöltése
            audio, sr = librosa.load(str(item['original']), sr=None)
            
            # Resample 16 kHz-re (ha kell)
            if sr != 16000:
                audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            else:
                audio_16k = audio
            
            # Új fájl mentése
            sf.write(str(item['output']), audio_16k, 16000)
            success += 1
            
        except Exception as e:
            print(f"\nWARNING: {item['original'].name}")
            print(f"   {str(e)}")
            failed += 1

    print(f"\nKész!")
    print(f"   Sikeres: {success} fájl")
    print(f"   Átugrott (már létezik): {skipped} fájl")
    print(f"   Sikertelen: {failed} fájl")
    print(f"   Új fájlok helye: {output_dir}")

if __name__ == "__main__":
    main()