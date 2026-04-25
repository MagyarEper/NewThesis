#!/usr/bin/env python3
"""
Generálási manifest készítése: new_texts.csv × összes speaker → generation_manifest.csv

Minden szöveget × minden speakerhez generál egy sort, így a Grad-TTS
generate_test_set.py automatikusan végigmegy az összes speaker-szöveg kombináción.

Használat:
    python create_generation_manifest.py \
        --texts new_texts.csv \
        --manifest manifest.csv \
        --output generation_manifest.csv \
        --wav-dir wavs_synthetic_80k

Kimenet:
    generation_manifest.csv  (columns: utt_id, wav, speaker, text)
"""

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Generálási manifest készítése")
    parser.add_argument("--texts", type=str, default="new_texts.csv",
                        help="Szöveg CSV fájl (default: new_texts.csv)")
    parser.add_argument("--manifest", type=str, default="manifest.csv",
                        help="Meglévő manifest a speaker lista kinyeréséhez (default: manifest.csv)")
    parser.add_argument("--output", type=str, default="generation_manifest.csv",
                        help="Output manifest fájl (default: generation_manifest.csv)")
    parser.add_argument("--wav-dir", type=str, default="wavs_synthetic_80k",
                        help="Output WAV könyvtár elérési útja (default: wavs_synthetic_80k)")
    parser.add_argument("--wav-dir-absolute", action="store_true",
                        help="Ha megadva, az --wav-dir abszolút útvonalként kezelendő. "
                             "Egyébként a script könyvtárához relatív.")
    args = parser.parse_args()

    print("=" * 60)
    print("Generálási manifest készítő")
    print("=" * 60)

    # 1. Szövegek betöltése
    texts_path = Path(args.texts)
    if not texts_path.exists():
        print(f"HIBA: Szöveg fájl nem található: {texts_path}", file=sys.stderr)
        sys.exit(1)

    texts_df = pd.read_csv(texts_path)
    required_cols = {"text_id", "text"}
    if not required_cols.issubset(texts_df.columns):
        print(f"HIBA: A szöveg CSV-nek tartalmaznia kell: {required_cols}", file=sys.stderr)
        sys.exit(1)

    texts_df = texts_df.dropna(subset=["text"])
    print(f"  Szövegek betöltve : {len(texts_df):,} db  ({texts_path})")

    # 2. Speaker lista kinyerése a manifestből
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"HIBA: Manifest fájl nem található: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    manifest_df = pd.read_csv(manifest_path)
    if "speaker" not in manifest_df.columns:
        print("HIBA: 'speaker' oszlop nem található a manifestben", file=sys.stderr)
        sys.exit(1)

    speakers = sorted(manifest_df["speaker"].dropna().unique())
    print(f"  Speakerek         : {len(speakers)} db  ({', '.join(str(s) for s in speakers[:5])}{'...' if len(speakers) > 5 else ''})")

    total_rows = len(texts_df) * len(speakers)
    print(f"  Összes sor        : {len(texts_df):,} × {len(speakers)} = {total_rows:,}")
    print(f"  Output            : {args.output}")
    print(f"  WAV könyvtár      : {args.wav_dir}/")
    print()

    # 3. WAV könyvtár útvonal
    if args.wav_dir_absolute:
        wav_base = Path(args.wav_dir)
    else:
        # Relatív útvonal: a script munkakönyvtárához képest
        wav_base = Path(args.wav_dir)

    # 4. Manifest generálása
    output_path = Path(args.output)
    rows_written = 0

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["utt_id", "wav", "speaker", "text"])

        for _, text_row in texts_df.iterrows():
            text_id = str(text_row["text_id"])
            text = str(text_row["text"]).strip()

            for speaker in speakers:
                utt_id = f"SYN_{speaker}_{text_id}"
                wav_path = str(wav_base / f"{utt_id}.wav")
                writer.writerow([utt_id, wav_path, speaker, text])
                rows_written += 1

    print(f"Kész! {rows_written:,} sor mentve → {output_path}")

    # 5. Ellenőrző preview
    print("\nElső néhány sor:")
    preview_df = pd.read_csv(output_path, nrows=5)
    print(preview_df.to_string(index=False))


if __name__ == "__main__":
    main()
