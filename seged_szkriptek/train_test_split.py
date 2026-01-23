#!/usr/bin/env python3
"""
Train/Valid/Test split készítése beszélőnként.

Használat:
    python train_test_split.py --manifest manifest.txt --output-dir Grad-TTS/resources/filelists/libri-tts/
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Train/Valid/Test split készítése')
    parser.add_argument('--manifest', type=str, default="manifest.txt",
                        help='A manifest fájl (default: manifest.txt)')
    parser.add_argument('--output-dir', type=str, 
                        default="Grad-TTS/resources/filelists/libri-tts",
                        help='Az output könyvtár (default: Grad-TTS/resources/filelists/libri-tts)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()
    
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    
    if not manifest_path.exists():
        print(f"❌ HIBA: A manifest fájl nem található: {manifest_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"📖 Manifest beolvasása: {manifest_path}")
    
    # Beolvasás
    data = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) == 3:
                data.append({'wav': parts[0], 'text': parts[1], 'speaker': int(parts[2])})

    df = pd.DataFrame(data)
    
    print(f"   Összesen: {len(df)} utterance")
    print(f"   Beszélők: {df['speaker'].nunique()}\n")

    train_list = []
    val_list = []
    test_list = []

    for speaker in sorted(df['speaker'].unique()):
        speaker_df = df[df['speaker'] == speaker]
        speaker_df = speaker_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        n_utterances = len(speaker_df)

        if n_utterances < 20:
            train_list.append(speaker_df)
        else:
            train_data, temp_data = train_test_split(speaker_df, test_size=0.2, random_state=args.seed)
            val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=args.seed)
            train_list.append(train_data)
            val_list.append(val_data)
            test_list.append(test_data)

    train_df = pd.concat(train_list).reset_index(drop=True)
    val_df = pd.concat(val_list).reset_index(drop=True) if val_list else pd.DataFrame(columns=df.columns)
    test_df = pd.concat(test_list).reset_index(drop=True) if test_list else pd.DataFrame(columns=df.columns)

    # Mentés
    def save_grad_format(df, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                f.write(f"{row['wav']}|{row['text']}|{row['speaker']}\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "train.txt"
    val_path = output_dir / "valid.txt"
    test_path = output_dir / "test.txt"
    
    save_grad_format(train_df, train_path)
    save_grad_format(val_df, val_path)
    save_grad_format(test_df, test_path)

    print("✅ Splitek létrehozva:")
    print(f"   Train: {train_path} ({len(train_df)} utterances)")
    print(f"   Valid: {val_path} ({len(val_df)} utterances)")
    print(f"   Test:  {test_path} ({len(test_df)} utterances)")
    print(f"\n📊 Beszélők száma:")
    print(f"   Train: {train_df['speaker'].nunique()}")
    print(f"   Valid: {val_df['speaker'].nunique()}")
    print(f"   Test:  {test_df['speaker'].nunique()}")

if __name__ == "__main__":
    main()
