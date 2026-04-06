#!/usr/bin/env python3
"""
Whisper fine-tune for Hungarian dysarthric speech recognition.

Based on SPECOM 2025 paper methodology (Leung et al.)
3 experiments:
  - Exp 1: Fine-tune on real dysarthric data
  - Exp 2: Fine-tune on synthetic (Grad-TTS) data only
  - Exp 3: Fine-tune on real + synthetic (data augmentation)

Usage:
    # Exp 1: Real data only
    python whisper_finetune.py \
        --train-manifest train_manifest.csv \
        --val-manifest val_manifest.csv \
        --experiment real \
        --output-dir whisper_finetuned/exp1_real

    # Exp 2: Synthetic data only
    python whisper_finetune.py \
        --train-manifest synthetic_train_manifest.csv \
        --val-manifest val_manifest.csv \
        --experiment synthetic \
        --output-dir whisper_finetuned/exp2_synthetic

    # Exp 3: Real + Synthetic (DAug)
    python whisper_finetune.py \
        --train-manifest train_manifest.csv \
        --synth-manifest synthetic_train_manifest.csv \
        --val-manifest val_manifest.csv \
        --experiment daug \
        --output-dir whisper_finetuned/exp3_daug
"""

import argparse
import os
import csv
import torch
import evaluate
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from pathlib import Path

from datasets import Dataset, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# ============================================================
# Data loading
# ============================================================

def load_manifest(manifest_path: str) -> List[dict]:
    """Load a CSV manifest file (utt_id,wav,speaker,text)."""
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "audio": row["wav"],
                "sentence": row["text"],
                "speaker": row["speaker"],
            })
    print(f"Loaded {len(rows)} utterances from {manifest_path}")
    return rows


def build_dataset(manifest_paths: List[str]) -> Dataset:
    """Build a HuggingFace Dataset from one or more manifest CSVs."""
    all_rows = []
    for p in manifest_paths:
        all_rows.extend(load_manifest(p))

    ds = Dataset.from_list(all_rows)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds


# ============================================================
# Data collator
# ============================================================

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Extract audio features
        input_features = [
            self.processor.feature_extractor(
                f["audio"]["array"],
                sampling_rate=f["audio"]["sampling_rate"],
            ).input_features[0]
            for f in features
        ]
        batch = {"input_features": torch.tensor(np.array(input_features))}

        # Tokenize labels
        label_features = [
            self.processor.tokenizer(f["sentence"]).input_ids
            for f in features
        ]
        # Pad labels
        max_label_len = max(len(l) for l in label_features)
        labels_padded = []
        for l in label_features:
            padded = l + [-100] * (max_label_len - len(l))
            labels_padded.append(padded)
        batch["labels"] = torch.tensor(labels_padded, dtype=torch.long)

        return batch


# ============================================================
# Metrics
# ============================================================

def make_compute_metrics(processor):
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    return compute_metrics


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Whisper fine-tune for Hungarian dysarthric ASR")
    parser.add_argument("--train-manifest", type=str, required=True,
                        help="Path to training manifest CSV (real or synthetic)")
    parser.add_argument("--synth-manifest", type=str, default=None,
                        help="Path to synthetic manifest CSV (for DAug experiment)")
    parser.add_argument("--val-manifest", type=str, required=True,
                        help="Path to validation manifest CSV")
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["real", "synthetic", "daug"],
                        help="Experiment type")
    parser.add_argument("--output-dir", type=str, default="whisper_finetuned",
                        help="Output directory for checkpoints")
    parser.add_argument("--model-name", type=str, default="openai/whisper-small",
                        help="Whisper model to fine-tune")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=2,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=500,
                        help="Warmup steps")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--no-lora", action="store_true",
                        help="Disable LoRA (full fine-tune)")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use FP16 mixed precision")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Whisper Fine-tune — Experiment: {args.experiment}")
    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")

    # ---- Load processor & model ----
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name)
    tokenizer = WhisperTokenizer.from_pretrained(args.model_name, language="hu", task="transcribe")
    processor = WhisperProcessor.from_pretrained(args.model_name, language="hu", task="transcribe")

    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    model.generation_config.language = "hu"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    # ---- LoRA ----
    if not args.no_lora:
        print(f"Applying LoRA (r={args.lora_r})...")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                            "fc1", "fc2"],
            lora_dropout=0.05,
            bias="none",
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        print("Full fine-tune (no LoRA)")

    # ---- Build datasets ----
    train_manifests = [args.train_manifest]
    if args.experiment == "daug" and args.synth_manifest:
        train_manifests.append(args.synth_manifest)
        print(f"DAug mode: combining {args.train_manifest} + {args.synth_manifest}")

    print("\nLoading training data...")
    train_dataset = build_dataset(train_manifests)
    print(f"Training samples: {len(train_dataset)}")

    print("Loading validation data...")
    val_dataset = build_dataset([args.val_manifest])
    print(f"Validation samples: {len(val_dataset)}")

    # ---- Data collator ----
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # ---- Training args ----
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=50,
        save_total_limit=3,
        report_to="tensorboard",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        label_names=["labels"],
        gradient_checkpointing=True,
    )

    # ---- Trainer ----
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(processor),
        tokenizer=processor.feature_extractor,
    )

    # ---- Train ----
    print("\nStarting training...")
    trainer.train()

    # ---- Save best model ----
    best_dir = os.path.join(args.output_dir, "best_model")
    trainer.save_model(best_dir)
    processor.save_pretrained(best_dir)
    print(f"\nBest model saved to {best_dir}")
    print(f"Best WER: {trainer.state.best_metric:.4f}")
    print("Done!")


if __name__ == "__main__":
    main()
