"""
Upload Model Weights to HuggingFace Hub
Author: Praruj Thapa

Run this script ONCE after creating your HuggingFace account and extracting the model zips.

Usage:
    pip install huggingface_hub
    huggingface-cli login          # enter your HF token
    python scripts/upload_models_to_hf.py

Prerequisites:
    1. Create account at https://huggingface.co
    2. Go to https://huggingface.co/settings/tokens → create token with Write access
    3. Run `huggingface-cli login` and paste the token
    4. Extract your model zips to a local folder first:
       - detr-resnet-50-hardhat-finetuned.zip  → ./model_weights/detr-hardhat/
       - oxford-pets-vit-from-scratch.zip       → ./model_weights/vit-base/
       - oxford-pets-vit-with-kd.zip            → ./model_weights/vit-kd/
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# ─── Configuration ─────────────────────────────────────────────────────────────
HF_USERNAME   = "PRaruj"   # ← Change to your HF username if different
MODEL_WEIGHTS = Path("./model_weights")  # ← Folder where you extracted the zips

REPOS = {
    "detr-hardhat":    (MODEL_WEIGHTS / "detr-hardhat",   f"{HF_USERNAME}/detr-resnet50-hardhat-finetuned"),
    "vit-base":        (MODEL_WEIGHTS / "vit-base",        f"{HF_USERNAME}/oxford-pets-vit-from-scratch"),
    "vit-kd":          (MODEL_WEIGHTS / "vit-kd",          f"{HF_USERNAME}/oxford-pets-vit-with-kd"),
}

api = HfApi()


def upload_model(local_path: Path, repo_id: str, model_name: str):
    print(f"\n{'='*60}")
    print(f"Uploading: {model_name}")
    print(f"  Source : {local_path}")
    print(f"  Repo   : https://huggingface.co/{repo_id}")
    print(f"{'='*60}")

    if not local_path.exists():
        print(f"⚠️  SKIP: {local_path} not found.")
        print(f"   Please extract the zip to {local_path} first.")
        return

    # Create the repo on HF Hub (public, model type)
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    print(f"✅ Repo created/confirmed: {repo_id}")

    # Upload all files in the folder
    api.upload_folder(
        folder_path=str(local_path),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload {model_name} weights",
    )
    print(f"✅ Upload complete! View at: https://huggingface.co/{repo_id}")


def main():
    print("\n🤗 HuggingFace Model Upload Script")
    print("=" * 60)
    print("Make sure you are logged in: `huggingface-cli login`")
    print()

    # Check which zips have been extracted
    for name, (local_path, repo_id) in REPOS.items():
        exists = "✅" if local_path.exists() else "❌ (not found — extract zip first)"
        print(f"  {exists}  {name}  →  {local_path}")

    print()
    proceed = input("Proceed with upload? (y/n): ").strip().lower()
    if proceed != "y":
        print("Aborted.")
        return

    for name, (local_path, repo_id) in REPOS.items():
        upload_model(local_path, repo_id, name)

    print("\n" + "=" * 60)
    print("🎉 All uploads complete!")
    print()
    print("Next steps:")
    print("  1. Go to https://huggingface.co/PRaruj and verify your models")
    print("  2. Create a Space at https://huggingface.co/new-space")
    print("     - SDK: Gradio")
    print("     - Upload demo/app.py and demo/requirements.txt")
    print("  3. Update README.md with your actual Space URL")


if __name__ == "__main__":
    main()
