#!/usr/bin/env python3
"""
Magyar mondatok szűrése szintetikus TTS generáláshoz.

Két forrást támogat:
  1. Wikipedia (alapértelmezett, nincs autentikáció szükséges)
  2. Helyi Common Voice TSV fájl (kézzel letöltött validated.tsv)

A Grad-TTS text pipeline unidecode-ot használ (ékezetek -> ASCII), ezért
szűrés után ellenőrizzük, hogy az unidecode output értelmesen megmarad.

Használat:
    # Wikipedia (autentikáció nélkül):
    python filter_texts.py --target 2100 --output new_texts.csv

    # Helyi Common Voice TSV:
    python filter_texts.py --tsv /path/to/validated.tsv --target 2100 --output new_texts.csv

Kimenet:
    new_texts.csv  (columns: text_id, text)
"""

import re
import argparse
import sys
import csv
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from unidecode import unidecode
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Szűrési paraméterek
# ---------------------------------------------------------------------------

ALLOWED_CHARS_RE = re.compile(r"^[a-záéíóöőüűA-ZÁÉÍÓÖŐÜŰ ,'\-]+$")
MIN_WORDS = 3
MAX_WORDS = 10


# ---------------------------------------------------------------------------
# Segédfüggvények
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Mondatvégi írásjeleket eltávolítja, középső írásjeleket (pont, kettőspont, pontosvessző) közönre csere."""
    text = text.strip()
    # Mondat végéről leállítja a mondatvégi írásjeleket
    text = re.sub(r"[.!?]+$", "", text).strip()
    # Belső írásjelek: kettőspont, pontosvessző, zárójelek eltávolítása
    text = re.sub(r'[;:()\[\]/\\"]', " ", text)
    # Több közön normalizálása
    text = " ".join(text.split())
    return text


def is_valid(text, min_words=MIN_WORDS, max_words=MAX_WORDS):
    text = text.strip()
    if not text:
        return False
    if re.search(r"\d", text):
        return False
    if not ALLOWED_CHARS_RE.match(text):
        return False
    word_count = len(text.split())
    if word_count < min_words or word_count > max_words:
        return False
    decoded = unidecode(text).strip()
    if not decoded or len(decoded) < 3:
        return False
    meaningful_chars = sum(1 for c in decoded if c.isalpha())
    if meaningful_chars < len(decoded) * 0.5:
        return False
    return True


def normalize(text):
    return " ".join(text.lower().split())


def load_existing_texts(manifest_path):
    existing = set()
    path = Path(manifest_path)
    if not path.exists():
        print(f"  FIGYELEM: Manifest nem talalhato: {manifest_path} -- deduplikacio kihagyva",
              file=sys.stderr)
        return existing
    df = pd.read_csv(path)
    if "text" not in df.columns:
        print("  FIGYELEM: 'text' oszlop nem talalhato a manifestben", file=sys.stderr)
        return existing
    for text in df["text"].dropna():
        existing.add(normalize(str(text)))
    print(f"  {len(existing)} egyedi szoveg betoltve a manifestbol (deduplikacihoz)")
    return existing


def iter_sentences_from_wikipedia():
    print("   Dataset: wikimedia/wikipedia / 20231101.hu (streaming, nincs auth)")
    dataset = load_dataset("wikimedia/wikipedia", "20231101.hu", split="train", streaming=True)
    for sample in dataset:
        article_text = sample.get("text", "")
        sentences = re.split(r"(?<=[.!?])\s+", article_text)
        for sentence in sentences:
            yield sentence.strip()


def iter_sentences_from_tsv(tsv_path):
    path = Path(tsv_path)
    if not path.exists():
        print(f"HIBA: TSV fajl nem talalhato: {tsv_path}", file=sys.stderr)
        sys.exit(1)
    print(f"   Fajl: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t", usecols=["sentence"], dtype=str)
    for sentence in df["sentence"].dropna():
        yield sentence.strip()


def collect_sentences(source_iter, existing_texts, target, min_words, max_words):
    collected = []
    seen_normalized = set(existing_texts)
    skipped_invalid = 0
    skipped_duplicate = 0
    total_seen = 0

    pbar = tqdm(source_iter, desc="Szures", unit=" mondat")
    for raw_text in pbar:
        total_seen += 1
        text = clean_text(raw_text)
        if not is_valid(text, min_words, max_words):
            skipped_invalid += 1
            continue
        norm = normalize(text)
        if norm in seen_normalized:
            skipped_duplicate += 1
            continue
        seen_normalized.add(norm)
        collected.append(text)
        pbar.set_postfix({"gyujtott": len(collected), "cel": target})
        if len(collected) >= target:
            break

    pbar.close()
    return collected, total_seen, skipped_invalid, skipped_duplicate


def main():
    parser = argparse.ArgumentParser(
        description="Magyar mondatok szurese TTS generalashoz",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Peldak:
  # Wikipedia (autentikacio nelkul):
  python filter_texts.py --target 2100 --output new_texts.csv

  # Kezzel letoltott Common Voice TSV:
  python filter_texts.py --tsv /path/to/hu/validated.tsv --target 2100
        """
    )
    parser.add_argument("--output", type=str, default="new_texts.csv")
    parser.add_argument("--target", type=int, default=2100)
    parser.add_argument("--manifest", type=str, default="manifest.csv")
    parser.add_argument("--tsv", type=str, default=None,
                        help="Helyi Common Voice validated.tsv. Ha nincs megadva, Wikipedia-t hasznalunk.")
    parser.add_argument("--min-words", type=int, default=MIN_WORDS)
    parser.add_argument("--max-words", type=int, default=MAX_WORDS)
    args = parser.parse_args()

    print("=" * 60)
    print("Magyar szovegszuro -- TTS generalashoz")
    print("=" * 60)
    source_name = f"Common Voice TSV: {args.tsv}" if args.tsv else "Wikipedia (wikimedia/wikipedia hu)"
    print(f"  Forras            : {source_name}")
    print(f"  Szoszam tartomany : {args.min_words}-{args.max_words}")
    print(f"  Cel mondatszam    : {args.target}")
    print(f"  Output            : {args.output}")
    print()

    print("1. Meglevo szovegek betoltese...")
    existing_texts = load_existing_texts(args.manifest)

    print("\n2. Forras betoltese...")
    if args.tsv:
        source_iter = iter_sentences_from_tsv(args.tsv)
    else:
        source_iter = iter_sentences_from_wikipedia()

    print("\n3. Mondatok szurese es gyujtese...")
    collected, total_seen, skipped_invalid, skipped_duplicate = collect_sentences(
        source_iter, existing_texts, args.target, args.min_words, args.max_words
    )

    print(f"\n  Osszesen feldolgozott : {total_seen:,}")
    print(f"  Ervenytelen/szurt    : {skipped_invalid:,}")
    print(f"  Duplikatum           : {skipped_duplicate:,}")
    print(f"  Gyujtott             : {len(collected):,}")

    if len(collected) < args.target:
        print(f"\n  FIGYELEM: Csak {len(collected)} mondatot sikerult gyujteni "
              f"(cel: {args.target}).", file=sys.stderr)

    if not collected:
        print("HIBA: Egyetlen mondat sem gyult ossze.", file=sys.stderr)
        sys.exit(1)

    word_counts = [len(t.split()) for t in collected]
    print(f"\n4. Szohossz-statisztikak:")
    print(f"   Min   : {min(word_counts)}")
    print(f"   Max   : {max(word_counts)}")
    print(f"   Atlag : {sum(word_counts) / len(word_counts):.1f}")

    print(f"\n5. Mentes: {args.output}")
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text_id", "text"])
        for i, text in enumerate(collected):
            writer.writerow([f"T{i:04d}", text])
    print(f"   Kesz: {len(collected)} mondat mentve -> {output_path}")
    print()

    print("Nehany pelda mondat:")
    for t in collected[:10]:
        print(f"  [{len(t.split())} szo] {t}  ->  {unidecode(t)}")


if __name__ == "__main__":
    main()
