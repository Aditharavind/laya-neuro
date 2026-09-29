#!/usr/bin/env python3
"""Push a fine-tuned laya-neuro checkpoint to the Hugging Face Hub.

Usage:
    python hf/push_model.py --repo-id Aditharavind/laya-neuro --checkpoint ../checkpoints/laya_neuro_cross_subject --card model_card_cross_subject.md
    python hf/push_model.py --repo-id Aditharavind/laya-neuro-within-subject --checkpoint ../checkpoints/laya_neuro_within_subject --card model_card_within_subject.md
"""
from __future__ import annotations

import argparse
import os

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--checkpoint", required=True, help="the fine-tuned checkpoint directory (train.py's --output-dir)")
    ap.add_argument("--card", required=True, help="model card filename in hf/")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    card_path = os.path.join(os.path.dirname(__file__), args.card)
    api.upload_file(path_or_fileobj=card_path, path_in_repo="README.md",
                     repo_id=args.repo_id, repo_type="model")

    api.upload_folder(folder_path=args.checkpoint, repo_id=args.repo_id, repo_type="model",
                       ignore_patterns=["checkpoint_latest/*", "checkpoint_latest"])
    print(f"done -> https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
