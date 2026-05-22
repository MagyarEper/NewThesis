#!/usr/bin/env python3
"""
Cross-speaker augmentáció manifest készítő.

Logika:
  - Minden speakerhez kiválasztjuk azokat a train mondatokat, amiket AZ A SPEAKER
    nem mondott el a train splitben
  - Kizárjuk a test halmazban is szereplő mondatokat (nincs leakage)
  - Speaker-enként ~N mondatot veszünk (összesen ~target_total)
  - Kimenet: generation_manifest_crossspeaker.csv  (utt_id, wav, speaker, text)

Használat:
    python create_crossspeaker_manifest.py \
        --train train_manifest.csv \
        --test test_manifest.csv \
        --output generation_manifest_crossspeaker.csv \
        --wav-dir wavs_synthetic_crossspeaker \
        --target 8000 \
        --seed 42
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def load_manifest(fname):
    rows = []
    with open(fname) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train_manifest.csv")
    parser.add_argument("--test", default="test_manifest.csv")
    parser.add_argument("--output", default="generation_manifest_crossspeaker.csv")
    parser.add_argument("--wav-dir", default="wavs_synthetic_crossspeaker")
    parser.add_argument("--target", type=int, default=8000,
                        help="Célzott összes generált utterance száma")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    train = load_manifest(args.train)
    test = load_manifest(args.test)

    # Összes train mondat szöveg szerint
    all_train_texts = sorted(set(r["text"].strip() for r in train))
    test_texts = set(r["text"].strip() for r in test)

    # Biztonságos forrás: train mondatok, amelyek NINCSENEK a tesztben
    safe_texts = sorted(set(all_train_texts) - test_texts)
    print(f"Összes egyedi train mondat:        {len(all_train_texts)}")
    print(f"Ebből test-mentes (biztonságos):   {len(safe_texts)}")

    # Per-speaker: mit mondott már el a train-ben
    said_by_speaker = defaultdict(set)
    for r in train:
        said_by_speaker[r["speaker"]].add(r["text"].strip())

    speakers = sorted(said_by_speaker.keys())
    per_speaker = args.target // len(speakers)
    print(f"Speakerek:                         {len(speakers)}")
    print(f"Cél/speaker:                       {per_speaker}")
    print()

    rows_out = []
    for sp in speakers:
        # Mondatok amiket ez a speaker NEM mondott, és teszt-mentes
        available = [t for t in safe_texts if t not in said_by_speaker[sp]]
        if len(available) < per_speaker:
            print(f"  FIGYELEM: {sp} — csak {len(available)} elérhető mondat (< {per_speaker})")
            selected = available
        else:
            selected = random.sample(available, per_speaker)

        for i, text in enumerate(selected):
            utt_id = f"XSPK_{sp}_T{i:04d}"
            wav_path = f"{args.wav_dir}/{utt_id}.wav"
            rows_out.append({
                "utt_id": utt_id,
                "wav": wav_path,
                "speaker": sp,
                "text": text,
            })

    # Kiírás
    output_path = Path(args.output)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["utt_id", "wav", "speaker", "text"])
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Összesen generált sorok: {len(rows_out)}")
    print(f"Output: {output_path}")

    # Leakage ellenőrzés
    out_texts = set(r["text"] for r in rows_out)
    leak = out_texts & test_texts
    print(f"Leakage ellenőrzés — teszt-overlap: {len(leak)} (kell: 0)")


if __name__ == "__main__":
    main()
