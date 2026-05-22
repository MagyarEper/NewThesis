#!/usr/bin/env python3
"""
Evaluate fine-tuned Whisper on Hungarian dysarthric test set.

Usage:
    # Baseline (no fine-tune)
    python whisper_evaluate.py \
        --test-manifest test_manifest.csv \
        --model-name openai/whisper-small \
        --output-csv results/exp0_baseline.csv

    # Fine-tuned model
    python whisper_evaluate.py \
        --test-manifest test_manifest.csv \
        --model-path whisper_finetuned/exp1_real/best_model \
        --output-csv results/exp1_real.csv
"""

import argparse
import csv
import os
import torch
import evaluate
import numpy as np
from pathlib import Path
from tqdm import tqdm

from datasets import Dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)
from peft import PeftModel


def load_manifest(manifest_path: str):
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "audio": row["wav"],
                "sentence": row["text"],
                "speaker": row["speaker"],
                "utt_id": row["utt_id"],
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper on dysarthric speech")
    parser.add_argument("--test-manifest", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="openai/whisper-small",
                        help="Base model name (for baseline or processor loading)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to fine-tuned model (None = use base model)")
    parser.add_argument("--output-csv", type=str, default="results/whisper_eval.csv")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load processor
    processor = WhisperProcessor.from_pretrained(args.model_name, language="hu", task="transcribe")

    # Load model
    if args.model_path:
        print(f"Loading fine-tuned model from {args.model_path}")
        adapter_config = Path(args.model_path) / "adapter_config.json"
        if adapter_config.exists():
            print("  Detected LoRA adapter — loading base model + adapter")
            model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
            model = PeftModel.from_pretrained(model, args.model_path)
            model = model.merge_and_unload()
        else:
            model = WhisperForConditionalGeneration.from_pretrained(args.model_path)
    else:
        print(f"Loading base model: {args.model_name}")
        model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

    model.generation_config.language = "hu"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model = model.to(device)
    model.eval()

    # Load test data
    rows = load_manifest(args.test_manifest)
    ds = Dataset.from_list(rows)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    # Run inference
    all_results = []
    all_preds = []
    all_refs = []

    print(f"\nRunning inference on {len(ds)} utterances...")
    for i in tqdm(range(len(ds))):
        sample = ds[i]
        audio = sample["audio"]["array"]
        ref_text = sample["sentence"]
        speaker = sample["speaker"]
        utt_id = sample["utt_id"]

        input_features = processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(input_features)

        pred_text = processor.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

        # Per-utterance WER + CER
        utt_wer = wer_metric.compute(predictions=[pred_text], references=[ref_text])
        utt_cer = cer_metric.compute(predictions=[pred_text], references=[ref_text])

        all_results.append({
            "utt_id": utt_id,
            "speaker": speaker,
            "reference": ref_text,
            "prediction": pred_text,
            "wer": utt_wer,
            "cer": utt_cer,
        })
        all_preds.append(pred_text)
        all_refs.append(ref_text)

    # Overall WER + CER
    overall_wer = wer_metric.compute(predictions=all_preds, references=all_refs)
    overall_cer = cer_metric.compute(predictions=all_preds, references=all_refs)

    # Per-speaker WER + CER
    speaker_preds = {}
    speaker_refs = {}
    for r in all_results:
        spk = r["speaker"]
        if spk not in speaker_preds:
            speaker_preds[spk] = []
            speaker_refs[spk] = []
        speaker_preds[spk].append(r["prediction"])
        speaker_refs[spk].append(r["reference"])

    speaker_wers = {}
    speaker_cers = {}
    for spk in sorted(speaker_preds.keys()):
        speaker_wers[spk] = wer_metric.compute(
            predictions=speaker_preds[spk], references=speaker_refs[spk]
        )
        speaker_cers[spk] = cer_metric.compute(
            predictions=speaker_preds[spk], references=speaker_refs[spk]
        )

    avg_speaker_wer = np.mean(list(speaker_wers.values()))
    avg_speaker_cer = np.mean(list(speaker_cers.values()))

    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Overall WER:         {overall_wer:.4f} ({overall_wer*100:.2f}%)")
    print(f"Overall CER:         {overall_cer:.4f} ({overall_cer*100:.2f}%)")
    print(f"Avg speaker WER:     {avg_speaker_wer:.4f} ({avg_speaker_wer*100:.2f}%)")
    print(f"Avg speaker CER:     {avg_speaker_cer:.4f} ({avg_speaker_cer*100:.2f}%)")
    print(f"\nPer-speaker WER / CER:")
    for spk, w in speaker_wers.items():
        n = len(speaker_preds[spk])
        c = speaker_cers[spk]
        print(f"  {spk}: WER={w:.4f} ({w*100:.1f}%)  CER={c:.4f} ({c*100:.1f}%)  [{n} utt]")

    # Save results
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["utt_id", "speaker", "reference", "prediction", "wer", "cer"])
        writer.writeheader()
        writer.writerows(all_results)

    # Save summary
    summary_path = args.output_csv.replace(".csv", "_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {args.model_path or args.model_name}\n")
        f.write(f"Test set: {args.test_manifest}\n")
        f.write(f"Utterances: {len(all_results)}\n")
        f.write(f"Overall WER: {overall_wer:.4f} ({overall_wer*100:.2f}%)\n")
        f.write(f"Overall CER: {overall_cer:.4f} ({overall_cer*100:.2f}%)\n")
        f.write(f"Avg speaker WER: {avg_speaker_wer:.4f} ({avg_speaker_wer*100:.2f}%)\n")
        f.write(f"Avg speaker CER: {avg_speaker_cer:.4f} ({avg_speaker_cer*100:.2f}%)\n\n")
        f.write("Per-speaker WER / CER:\n")
        for spk, w in speaker_wers.items():
            c = speaker_cers[spk]
            f.write(f"  {spk}: WER={w:.4f} ({w*100:.1f}%)  CER={c:.4f} ({c*100:.1f}%)\n")

    print(f"\nResults saved to {args.output_csv}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
