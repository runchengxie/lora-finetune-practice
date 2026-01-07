#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Upload LoRA training artifacts to the Hugging Face Hub (no CLI)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, login

DEFAULT_LOCAL_DIR = "outputs/SmolLM2-FT-MyDataset"
DEFAULT_COMMIT_MESSAGE = "Upload LoRA training artifacts"


def get_hf_token() -> str:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload LoRA training artifacts to the Hugging Face Hub."
    )
    parser.add_argument(
        "--local-dir",
        default=DEFAULT_LOCAL_DIR,
        help="Path to the training output directory.",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Destination repo, e.g. username/model-name.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create a public repo (default: private).",
    )
    parser.add_argument(
        "--exclude-checkpoints",
        action="store_true",
        help="Skip checkpoint-* directories to save space.",
    )
    parser.add_argument(
        "--commit-message",
        default=DEFAULT_COMMIT_MESSAGE,
        help="Commit message for the upload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = get_hf_token()
    if not token:
        raise SystemExit("HF_TOKEN is missing. Export HF_TOKEN=... first.")

    local_dir = Path(args.local_dir)
    if not local_dir.exists():
        raise SystemExit(f"Local directory not found: {local_dir}")

    login(token=token, add_to_git_credential=False)
    api = HfApi()

    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=not args.public,
        exist_ok=True,
    )

    ignore_patterns = []
    if args.exclude_checkpoints:
        ignore_patterns.extend(
            [
                "checkpoint-*",
                "**/checkpoint-*",
            ]
        )

    upload_kwargs = dict(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(local_dir),
        commit_message=args.commit_message,
    )
    if ignore_patterns:
        upload_kwargs["ignore_patterns"] = ignore_patterns

    api.upload_folder(**upload_kwargs)

    print(f"Done: uploaded {local_dir} -> {args.repo_id}")


if __name__ == "__main__":
    main()
