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


def get_repo_id() -> str:
    repo_id = os.environ.get("HF_REPO_ID") or ""
    if repo_id:
        return repo_id
    username = os.environ.get("HF_USERNAME") or ""
    repo_name = os.environ.get("HF_REPO_NAME") or ""
    if username and repo_name:
        return f"{username}/{repo_name}"
    return ""


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
        default=None,
        help=(
            "Destination repo, e.g. username/model-name. "
            "Defaults to HF_REPO_ID or HF_USERNAME + HF_REPO_NAME."
        ),
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

    repo_id = args.repo_id or get_repo_id()
    if not repo_id:
        raise SystemExit(
            "Repo id is missing. Pass --repo-id or set HF_REPO_ID "
            "(or HF_USERNAME + HF_REPO_NAME)."
        )

    local_dir = Path(args.local_dir)
    if not local_dir.exists():
        raise SystemExit(f"Local directory not found: {local_dir}")

    login(token=token, add_to_git_credential=False)
    api = HfApi()

    api.create_repo(
        repo_id=repo_id,
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
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(local_dir),
        commit_message=args.commit_message,
    )
    if ignore_patterns:
        upload_kwargs["ignore_patterns"] = ignore_patterns

    api.upload_folder(**upload_kwargs)

    print(f"Done: uploaded {local_dir} -> {repo_id}")


if __name__ == "__main__":
    main()
