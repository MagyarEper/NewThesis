#!/usr/bin/env python3
"""
Szöveg-diszjunkt kiértékelés: csak azokat a teszt utterance-öket értékeli ki,
amelyek szövege NEM szerepel a tanítóhalmazban.

Ez megmutatja, hogy a modell teljesítménye mennyire függ a szöveg-memorization-től
versus valódi akusztikai adaptációtól.

Használat:
    python evaluate_unseen_texts.py \
        --model-path whisper_finetuned/exp1_real/best_model \
        --output-csv results/exp1_unseen.csv

    # Baseline
    python evaluate_unseen_texts.py \
        --model-name openai/whisper-small \
        --output-csv results/exp0_unseen.csv
"""

import argparse
import csv
import os
import re
import torch
import evaluate
import numpy as np
from pathlib import Path
from tqdm import tqdm

from datasets import Dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperConfig
from peft import PeftModel


TRAIN_MANIFEST = "train_manifest.csv"
TEST_MANIFEST  = "test_manifest.csv"


def normalize(text: str) -> str:
    """Azonos normalizálás mint a whisper_evaluate.py-ban."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return text.strip()


def load_manifest(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_unseen_test(train_manifest: str, test_manifest: str):
    train_rows = load_manifest(train_manifest)
    test_rows  = load_manifest(test_manifest)

    train_texts = set(normalize(r["text"]) for r in train_rows)

    unseen = [r for r in test_rows if normalize(r["text"]) not in train_texts]
    seen   = [r for r in test_rows if normalize(r["text"]) in train_texts]

    print(f"Teszt összesen:          {len(test_rows)}")
    print(f"  Szöveg látott (train): {len(seen)}  ({100*len(seen)/len(test_rows):.1f}%)")
    print(f"  Szöveg NEM látott:     {len(unseen)}  ({100*len(unseen)/len(test_rows):.1f}%)")
    return unseen


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-path", type=str,
                       help="LoRA checkpoint könyvtár (adapter_config.json)")
    group.add_argument("--model-name", type=str,
                       help="HuggingFace model ID (pl. openai/whisper-small)")
    parser.add_argument("--output-csv", type=str, required=True,
                        help="Kimeneti CSV fájl")
    parser.add_argument("--train-manifest", type=str, default=TRAIN_MANIFEST)
    parser.add_argument("--test-manifest",  type=str, default=TEST_MANIFEST)
    args = parser.parse_args()

    # --- Unseen szűrés ---
    print("=" * 60)
    print("Szöveg-diszjunkt kiértékelés")
    print("=" * 60)
    unseen_rows = build_unseen_test(args.train_manifest, args.test_manifest)

    if len(unseen_rows) == 0:
        print("FIGYELEM: Nincs egyetlen nem látott szöveg sem a teszthalmazban!")
        return

    # --- Modell betöltés ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nEszköz: {device}")

    BASE_MODEL = "openai/whisper-small"
    # Processor is never modified during fine-tuning; always load from base model
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)

    if args.model_path:
        model_dir = Path(args.model_path).resolve()
        is_lora = (model_dir / "adapter_config.json").exists()
        if is_lora:
            print(f"LoRA adapter betöltése: {model_dir}")
            base = WhisperForConditionalGeneration.from_pretrained(
                BASE_MODEL, torch_dtype=torch.float16
            ).to(device)
            model = PeftModel.from_pretrained(base, str(model_dir))
            model = model.merge_and_unload()
        else:
            print(f"Teljes modell betöltése: {model_dir}")
            # Bypass from_pretrained entirely to avoid HF hub validation on local paths
            config = WhisperConfig.from_json_file(str(model_dir / "config.json"))
            model = WhisperForConditionalGeneration(config)
            st_path = model_dir / "model.safetensors"
            pt_path = model_dir / "pytorch_model.bin"
            if st_path.exists():
                from safetensors.torch import load_file
                sd = load_file(str(st_path), device="cpu")
            else:
                sd = torch.load(str(pt_path), map_location="cpu")
            model.load_state_dict(sd)
            model = model.to(dtype=torch.float16, device=device)
    else:
        print(f"Alap modell betöltése: {args.model_name}")
        model = WhisperForConditionalGeneration.from_pretrained(
            args.model_name, torch_dtype=torch.float16
        ).to(device)

    model.eval()
    forced_ids = processor.get_decoder_prompt_ids(language="hu", task="transcribe")
    model.config.forced_decoder_ids = forced_ids

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    # --- Dataset ---
    hf_rows = [{"audio": r["wav"], "sentence": r["text"],
                 "speaker": r["speaker"], "utt_id": r["utt_id"]}
                for r in unseen_rows]
    dataset = Dataset.from_list(hf_rows).cast_column("audio", Audio(sampling_rate=16000))

    # --- Kiértékelés ---
    results = []
    all_hyp, all_ref = [], []
    speaker_data = {}

    print(f"\nKiértékelés {len(unseen_rows)} nem látott utterance-en...\n")
    for sample in tqdm(dataset):
        audio_array = sample["audio"]["array"]
        ref = normalize(sample["sentence"])
        speaker = sample["speaker"]
        utt_id  = sample["utt_id"]

        inputs = processor(
            audio_array, sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(device).half()

        with torch.no_grad():
            pred_ids = model.generate(inputs, forced_decoder_ids=forced_ids)
        hyp = normalize(processor.batch_decode(pred_ids, skip_special_tokens=True)[0])

        utt_wer = wer_metric.compute(predictions=[hyp], references=[ref])
        utt_cer = cer_metric.compute(predictions=[hyp], references=[ref])

        results.append({
            "utt_id": utt_id, "speaker": speaker,
            "reference": ref, "hypothesis": hyp,
            "wer": utt_wer, "cer": utt_cer
        })
        all_hyp.append(hyp)
        all_ref.append(ref)

        if speaker not in speaker_data:
            speaker_data[speaker] = {"hyp": [], "ref": [], "count": 0}
        speaker_data[speaker]["hyp"].append(hyp)
        speaker_data[speaker]["ref"].append(ref)
        speaker_data[speaker]["count"] += 1

    # --- Összesítés ---
    overall_wer = wer_metric.compute(predictions=all_hyp, references=all_ref)
    overall_cer = cer_metric.compute(predictions=all_hyp, references=all_ref)
    sp_wers = [wer_metric.compute(predictions=v["hyp"], references=v["ref"])
               for v in speaker_data.values()]
    avg_sp_wer = float(np.mean(sp_wers))

    print(f"\nEredmények (csak nem látott szövegek):")
    print(f"  Utterances:       {len(results)}")
    print(f"  Overall WER:      {overall_wer:.4f} ({100*overall_wer:.2f}%)")
    print(f"  Overall CER:      {overall_cer:.4f} ({100*overall_cer:.2f}%)")
    print(f"  Avg speaker WER:  {avg_sp_wer:.4f} ({100*avg_sp_wer:.2f}%)")

    # --- CSV mentés ---
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["utt_id","speaker","reference","hypothesis","wer","cer"])
        writer.writeheader()
        writer.writerows(results)

    # --- Summary ---
    summary_path = out_path.with_suffix("").as_posix() + "_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {args.model_path or args.model_name}\n")
        f.write(f"Test set: {args.test_manifest} (csak nem látott szövegek)\n")
        f.write(f"Utterances: {len(results)}\n")
        f.write(f"Overall WER: {overall_wer:.4f} ({100*overall_wer:.2f}%)\n")
        f.write(f"Overall CER: {overall_cer:.4f} ({100*overall_cer:.2f}%)\n")
        f.write(f"Avg speaker WER: {avg_sp_wer:.4f} ({100*avg_sp_wer:.2f}%)\n")
        f.write("\nPer-speaker WER:\n")
        for sp, v in sorted(speaker_data.items()):
            sp_wer = wer_metric.compute(predictions=v["hyp"], references=v["ref"])
            f.write(f"  {sp}: {sp_wer:.4f} ({100*sp_wer:.1f}%)  [{v['count']} utt]\n")

    print(f"\nEredmények mentve: {out_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
