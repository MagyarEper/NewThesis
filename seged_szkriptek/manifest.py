#!/usr/bin/env python3
"""
Manifest létrehozása GRAD-TTS-hez.

Használat:
    python manifest.py --xlsx <transcript.xlsx> --output manifest.txt --wav-dir wavs_16khz/
"""

import re
import pandas as pd
from pathlib import Path
import argparse
import sys

def remove_brackets(text):
    if pd.isna(text):
        return text
    return re.sub(r'\[|\]', '', text)

def main():
    parser = argparse.ArgumentParser(description='Manifest létrehozása GRAD-TTS-hez')
    parser.add_argument('--xlsx', type=str, required=True,
                        help='Az xlsx transcript fájl elérési útja')
    parser.add_argument('--output', type=str, default="manifest.txt",
                        help='A manifest output fájl (default: manifest.txt)')
    parser.add_argument('--wav-dir', type=str, default="wavs_16khz",
                        help='A WAV könyvtár (default: wavs_16khz)')

    args = parser.parse_args()

    xlsx_path = Path(args.xlsx)
    output_txt = Path(args.output)
    wav_base_dir = Path(args.wav_dir)
    
    if not xlsx_path.exists():
        print(f"❌ HIBA: Az xlsx fájl nem található: {xlsx_path}", file=sys.stderr)
        sys.exit(1)
    
    if not wav_base_dir.exists():
        print(f"⚠️  FIGYELEM: A WAV könyvtár nem létezik: {wav_base_dir}")
    
    print(f"📖 Transcript beolvasása: {xlsx_path}")
    df = pd.read_excel(xlsx_path)

    required_cols = ['ID', 'Full_ID', 'Transcript']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ HIBA: Hiányzó oszlopok: {missing_cols}", file=sys.stderr)
        sys.exit(1)

    speakers = sorted(df['ID'].unique())
    speaker_map = {spk: idx for idx, spk in enumerate(speakers)}

    print(f"\n👥 Speaker mapping ({len(speaker_map)} speaker):")
    for spk, idx in sorted(speaker_map.items())[:5]:
        print(f"  {spk} -> {idx}")
    if len(speaker_map) > 5:
        print(f"  ...")

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📝 Manifest írása: {output_txt}")
    with open(output_txt, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            wav_path = wav_base_dir / f"{row['Full_ID']}.wav"
            text = remove_brackets(row['Transcript'])
            speaker_id = speaker_map[row['ID']]
            line = f"{wav_path}|{text}|{speaker_id}\n"
            f.write(line)

    print(f"\n✅ Manifest létrehozva: {output_txt}")
    print(f"   Sorok: {len(df)}, Beszélők: {len(speaker_map)}")

if __name__ == "__main__":
    main()
