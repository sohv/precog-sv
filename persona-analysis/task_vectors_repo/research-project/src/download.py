# src/download.py

import os
from huggingface_hub import snapshot_download

def download_base_model(model_name: str, local_path: str):

    # Check if the directory exists and is not empty
    if os.path.exists(local_path) and os.listdir(local_path):
        print(f"Base model '{model_name}' already found at '{local_path}'. Skipping download.")
        return

    print(f"Downloading base model '{model_name}' to '{local_path}'...")
    
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=local_path,
            local_dir_use_symlinks=False, # Set to False to copy files instead of symlinking
            resume_download=True
        )
        print(f"Successfully downloaded {model_name}.")
    except Exception as e:
        print(f"Failed to download model {model_name}. Error: {e}")
        raise