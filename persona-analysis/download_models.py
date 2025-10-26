import os
import argparse
from huggingface_hub import snapshot_download

huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir /scratch/manas/Llama-3.1-8B-Instruct/ --local-dir-use-symlinks False

def main():
    parser = argparse.ArgumentParser(description="Download a model from Hugging Face Hub to a specified directory.")
    parser.add_argument("--model_id", type=str, required=True, help="The model ID to download (e.g., 'meta-llama/Llama-3.1-8B-Instruct').")
    parser.add_argument("--save_path", type=str, required=True, help="The local directory to save the model to.")
    args = parser.parse_args()

    # Ensure the save path exists
    os.makedirs(args.save_path, exist_ok=True)

    print(f"Downloading model '{args.model_id}' to '{args.save_path}'...")
    
    # This is the function that does all the work.
    # `local_dir_use_symlinks=False` is important for some cluster filesystems.
    snapshot_download(
        repo_id=args.model_id,
        local_dir=args.save_path,
        local_dir_use_symlinks=False
    )

    print("Download complete!")

if __name__ == "__main__":
    main()