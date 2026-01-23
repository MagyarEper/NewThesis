#!/usr/bin/env python3
"""
WAV fájlok ellenőrzése és hiányzó fájlok eltávolítása a manifestből.

Használat:
    python check_files.py --manifest manifest.txt
"""

import sys
from pathlib import Path
from collections import Counter
import argparse

def main():
    parser = argparse.ArgumentParser(description='WAV fájlok ellenőrzése')
    parser.add_argument('--manifest', type=str, default="manifest.txt",
                        help='A manifest fájl elérési útja (default: manifest.txt)')
    args = parser.parse_args()
    
    manifest_path = Path(args.manifest)
    
    if not manifest_path.exists():
        print(f"❌ HIBA: A manifest fájl nem található: {manifest_path}", file=sys.stderr)
        sys.exit(1)
    
    # Manifest beolvasása
    data = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) == 3:
                data.append({'wav': parts[0], 'text': parts[1], 'speaker': parts[2]})

    print(f"📊 Összesen {len(data)} fájl a manifestben\n")

    # Ellenőrzés
    missing = []
    existing = []

    for item in data:
        wav_path = Path(item['wav'])
        if wav_path.exists():
            existing.append(item)
        else:
            missing.append(item)

    print(f"✓ Megtalált fájlok: {len(existing)}")
    print(f"✗ Hiányzó fájlok: {len(missing)}\n")

    if len(missing) > 50:
        print("❌ HIBA: Több mint 50 fájl hiányzik!")
        print(f"   Hiányzó fájlok száma: {len(missing)}")
        print("\nHiányzó fájlok (első 50):")
        for i, item in enumerate(missing[:50], 1):
            print(f"  {i}. {item['wav']}")
        if len(missing) > 50:
            print(f"  ... és még {len(missing) - 50} további")
        sys.exit(1)

    if missing:
        print("Hiányzó fájlok listája:")
        for i, item in enumerate(missing, 1):
            print(f"  {i}. {item['wav']}")
        
        print(f"\n🔧 Hiányzó fájlok eltávolítása a manifestből...")
        
        with open(manifest_path, 'w', encoding='utf-8') as f:
            for item in existing:
                f.write(f"{item['wav']}|{item['text']}|{item['speaker']}\n")
        
        print(f"✅ Manifest frissítve!")
        print(f"   Eredeti: {len(data)} fájl")
        print(f"   Tisztított: {len(existing)} fájl")
        print(f"   Eltávolítva: {len(missing)} fájl")
    else:
        print("🎉 Minden fájl megvan!")

    # Statisztikák
    print("\n" + "="*50)
    print("Statisztikák beszélőnként:")
    print("="*50)

    final_data = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) == 3:
                final_data.append({'speaker': parts[2]})

    speaker_counts = Counter([item['speaker'] for item in final_data])
    for speaker, count in sorted(speaker_counts.items(), key=lambda x: int(x[0])):
        print(f"Speaker {speaker.zfill(2)}: {count} fájl")

if __name__ == "__main__":
    main()
