#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LoRA fine-tune practice for SmolLM2 with low-VRAM friendly defaults."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional

import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


@dataclass
class TrainSettings:
    model_name: str = "HuggingFaceTB/SmolLM2-135M"
    chat_template_path: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    dataset_path: str = "HuggingFaceTB/smoltalk"
    dataset_name: str = "everyday-conversations"
    output_dir: str = "SmolLM2-FT-MyDataset"
    max_seq_length: int = 512
    train_subset_size: int = 2000
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 1
    learning_rate: float = 2e-4
    logging_steps: int = 10
    save_strategy: str = "epoch"
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch"
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "constant"
    lora_r: int = 6
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    packing: bool = False
    merge_adapter: bool = False
    run_inference: bool = False
    skip_login: bool = False


def resolve_device() -> str:
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        arch = f"sm_{major}{minor}"
        try:
            arch_list = torch.cuda.get_arch_list()
        except Exception:
            arch_list = []
        if arch_list and not any(entry.startswith(arch) for entry in arch_list):
            print(
                "[warn] CUDA device capability "
                f"{major}.{minor} ({arch}) is not supported by this PyTorch build; "
                "falling back to CPU."
            )
        else:
            return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_hf_token() -> Optional[str]:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def maybe_login(skip_login: bool) -> None:
    if skip_login:
        return
    token = get_hf_token()
    if token:
        login(token=token, add_to_git_credential=False)


def build_peft_config(settings: TrainSettings) -> LoraConfig:
    return LoraConfig(
        r=settings.lora_r,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )


def build_sft_config(settings: TrainSettings, device: str) -> SFTConfig:
    base_kwargs = dict(
        output_dir=settings.output_dir,
        num_train_epochs=settings.num_train_epochs,
        per_device_train_batch_size=settings.per_device_train_batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        gradient_checkpointing=settings.gradient_checkpointing,
        optim=settings.optim,
        learning_rate=settings.learning_rate,
        max_grad_norm=settings.max_grad_norm,
        warmup_ratio=settings.warmup_ratio,
        lr_scheduler_type=settings.lr_scheduler_type,
        logging_steps=settings.logging_steps,
        save_strategy=settings.save_strategy,
        bf16=False,
        fp16=device == "cuda",
        push_to_hub=False,
        report_to="none",
        packing=settings.packing,
        max_seq_length=settings.max_seq_length,
        dataset_kwargs={"add_special_tokens": False, "append_concat_token": False},
    )
    sft_kwargs = dict(base_kwargs, chat_template_path=settings.chat_template_path)
    while True:
        try:
            return SFTConfig(**sft_kwargs)
        except TypeError as exc:
            match = re.search(r"unexpected keyword argument[s]?:? '([^']+)'", str(exc))
            if not match:
                raise
            bad_key = match.group(1)
            if bad_key == "max_seq_length" and "max_seq_length" in sft_kwargs:
                sft_kwargs.pop("max_seq_length", None)
                sft_kwargs["max_length"] = settings.max_seq_length
                continue
            if bad_key in sft_kwargs:
                sft_kwargs.pop(bad_key)
                continue
            raise


def load_model_and_tokenizer(settings: TrainSettings, device: str):
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        settings.model_name,
        torch_dtype=torch_dtype,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_training_dataset(settings: TrainSettings):
    dataset = load_dataset(settings.dataset_path, settings.dataset_name)
    train_dataset = dataset["train"]
    if settings.train_subset_size and settings.train_subset_size > 0:
        subset_size = min(len(train_dataset), settings.train_subset_size)
        train_dataset = train_dataset.select(range(subset_size))
    return train_dataset


def build_trainer(
    settings: TrainSettings,
    model,
    tokenizer,
    train_dataset,
    sft_config: SFTConfig,
    peft_config: LoraConfig,
) -> SFTTrainer:
    base_kwargs = dict(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )
    try:
        return SFTTrainer(processing_class=tokenizer, **base_kwargs)
    except TypeError:
        return SFTTrainer(tokenizer=tokenizer, **base_kwargs)


def merge_adapter(output_dir: str) -> None:
    model = AutoPeftModelForCausalLM.from_pretrained(
        output_dir,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="2GB",
    )


def run_inference_samples(model, tokenizer) -> None:
    prompts: List[str] = [
        "What is the capital of Germany? Explain why thats the case and if it was different in the past?",
        "Write a Python function to calculate the factorial of a number.",
        "A rectangular garden has a length of 25 feet and a width of 15 feet. If you want to build a fence around the entire garden, how many feet of fencing will you need?",
        "What is the difference between a fruit and a vegetable? Give examples of each.",
    ]

    model.eval()
    for prompt in prompts:
        if hasattr(tokenizer, "apply_chat_template"):
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted = prompt
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = text[len(formatted) :].strip() if text.startswith(formatted) else text
        print(f"    prompt:\n{prompt}")
        print(f"    response:\n{response}")
        print("-" * 50)


def main() -> None:
    settings = TrainSettings()
    device = resolve_device()

    maybe_login(settings.skip_login)
    train_dataset = load_training_dataset(settings)
    model, tokenizer = load_model_and_tokenizer(settings, device)
    peft_config = build_peft_config(settings)
    sft_config = build_sft_config(settings, device)

    trainer = build_trainer(
        settings,
        model,
        tokenizer,
        train_dataset,
        sft_config,
        peft_config,
    )

    trainer.train()
    trainer.save_model()

    if settings.merge_adapter:
        merge_adapter(settings.output_dir)

    if settings.run_inference:
        run_inference_samples(trainer.model, tokenizer)


if __name__ == "__main__":
    main()
