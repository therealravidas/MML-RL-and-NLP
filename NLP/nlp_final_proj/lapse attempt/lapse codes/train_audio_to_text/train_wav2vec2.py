#!/usr/bin/env python3
"""
Memory-safe Wav2Vec2 fine-tuning script.

Usage:
python train_wav2vec2_safe.py \
  --train_json /path/to/train.jsonl \
  --valid_json /path/to/validation.jsonl \
  --output_dir ./wav2vec2_out \
  --model_name facebook/wav2vec2-base-960h \
  --train_bs 1 --eval_bs 1 --grad_accum 8

This script is conservative to avoid std::bad_alloc.
"""
import argparse
import os
from datasets import load_dataset
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
)
import torch
from lazy_collator import LazyCollator
from compute_metrics import make_compute_metrics
import subprocess

def maybe_create_swap(size_str):
    # create swap only if really needed
    try:
        print("Creating swapfile of", size_str)
        subprocess.check_call(["sudo", "fallocate", "-l", size_str, "/swapfile"])
        subprocess.check_call(["sudo", "chmod", "600", "/swapfile"])
        subprocess.check_call(["sudo", "mkswap", "/swapfile"])
        subprocess.check_call(["sudo", "swapon", "/swapfile"])
        print("Swap created: /swapfile")
    except Exception as e:
        print("Swap creation failed (need sudo and fallocate):", e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--valid_json", required=True)
    parser.add_argument("--test_json", default=None)
    parser.add_argument("--model_name", default="facebook/wav2vec2-base-960h")
    parser.add_argument("--output_dir", default="./wav2vec2_output")
    parser.add_argument("--train_bs", type=int, default=1)   # conservative default
    parser.add_argument("--eval_bs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--create_swap", type=str, default=None,
                        help="Optional: create swap, value e.g. '8G' (requires sudo)")
    args = parser.parse_args()

    if args.create_swap:
        maybe_create_swap(args.create_swap)

    print("Loading metadata (JSON)...")
    data_files = {"train": args.train_json, "validation": args.valid_json}
    ds = load_dataset("json", data_files=data_files)
    print(ds)

    print("Loading model and processor:", args.model_name)
    processor = Wav2Vec2Processor.from_pretrained(args.model_name)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_name)

    # if no pad token, add it and resize embeddings
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(processor.tokenizer))

    # enable gradient checkpointing to save memory (trades compute)
    try:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing ENABLED")
    except Exception:
        print("Gradient checkpointing not supported for this model/version")

    # set up collator and metrics
    collator = LazyCollator(processor)
    compute_metrics = make_compute_metrics(processor)

    # TrainingArguments with conservative memory settings
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        evaluation_strategy="steps",
        eval_steps=2000,
        logging_steps=200,
        save_steps=2000,
        num_train_epochs=args.epochs,
        fp16=torch.cuda.is_available(),  # only use fp16 if CUDA is available
        learning_rate=args.lr,
        save_total_limit=3,
        push_to_hub=False,
        dataloader_num_workers=args.dataloader_num_workers,  # avoid multi-worker memory spikes
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
    )

    print("Starting training. Monitor memory with `watch -n1 free -h` or `htop`.")
    trainer.train()

    if args.test_json:
        print("Evaluating on test set...")
        test_ds = load_dataset("json", data_files={"test": args.test_json})["test"]
        metrics = trainer.evaluate(test_dataset=test_ds)
        print("Test metrics:", metrics)

if __name__ == "__main__":
    main()
