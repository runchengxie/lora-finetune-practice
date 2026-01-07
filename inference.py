#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run inference with a LoRA-finetuned model checkpoint."""

from __future__ import annotations

import argparse
from typing import List, Optional

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


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
    return "cpu"


def load_tokenizer(model_dir: str, fallback_model: Optional[str]) -> AutoTokenizer:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except OSError:
        if not fallback_model:
            raise
        print(
            "[warn] Tokenizer files not found in model directory; "
            "falling back to base model tokenizer."
        )
        tokenizer = AutoTokenizer.from_pretrained(fallback_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def format_prompt(tokenizer: AutoTokenizer, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def build_generate_kwargs(
    max_new_tokens: int,
    temperature: Optional[float],
    top_p: Optional[float],
) -> dict:
    kwargs = {"max_new_tokens": max_new_tokens}
    if temperature is not None:
        kwargs["temperature"] = temperature
        kwargs["do_sample"] = True
    if top_p is not None:
        kwargs["top_p"] = top_p
        kwargs["do_sample"] = True
    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference with a LoRA adapter saved in an output directory."
    )
    parser.add_argument(
        "--model-dir",
        default="outputs/SmolLM2-FT-MyDataset",
        help="Path to the fine-tuned output directory.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Fallback tokenizer source if model-dir lacks tokenizer files.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        help="Prompt text. Repeat this flag to send multiple prompts.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Force a device or let the script decide automatically.",
    )
    args = parser.parse_args()

    device = resolve_device() if args.device == "auto" else args.device
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = load_tokenizer(args.model_dir, args.base_model)
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    prompts: List[str] = args.prompt or [
        "请用一句话解释什么是LoRA。",
    ]

    generate_kwargs = build_generate_kwargs(
        args.max_new_tokens,
        args.temperature,
        args.top_p,
    )

    for prompt in prompts:
        formatted = format_prompt(tokenizer, prompt)
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, **generate_kwargs)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = text[len(formatted) :].strip() if text.startswith(formatted) else text
        print(f"prompt:\n{prompt}")
        print(f"response:\n{response}")
        print("-" * 50)


if __name__ == "__main__":
    main()
