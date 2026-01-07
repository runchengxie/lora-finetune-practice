#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Upload LoRA training artifacts to the Hugging Face Hub (no CLI)."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, login

LOCAL_DIR = "SmolLM2-FT-MyDataset"  # training output directory
REPO_ID = "your-username/your-repo"  # example: richard/smollm2-lora-mydataset
PRIVATE = True
INCLUDE_CHECKPOINTS = True


def get_hf_token() -> str:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or ""


def main() -> None:
    token = get_hf_token()
    if not token:
        raise SystemExit("HF_TOKEN is missing. Export HF_TOKEN=... first.")

    local_dir = Path(LOCAL_DIR)
    if not local_dir.exists():
        raise SystemExit(f"Local directory not found: {local_dir}")

    login(token=token, add_to_git_credential=False)
    api = HfApi()

    api.create_repo(
        repo_id=REPO_ID,
        repo_type="model",
        private=PRIVATE,
        exist_ok=True,
    )

    ignore_patterns = []
    if not INCLUDE_CHECKPOINTS:
        ignore_patterns.extend(
            [
                "checkpoint-*",
                "**/checkpoint-*",
            ]
        )

    upload_kwargs = dict(
        repo_id=REPO_ID,
        repo_type="model",
        folder_path=str(local_dir),
        commit_message="Upload LoRA training artifacts",
    )
    if ignore_patterns:
        upload_kwargs["ignore_patterns"] = ignore_patterns

    api.upload_folder(**upload_kwargs)

    print(f"Done: uploaded {local_dir} -> {REPO_ID}")


if __name__ == "__main__":
    main()
