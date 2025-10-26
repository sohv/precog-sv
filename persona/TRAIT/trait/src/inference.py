import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from huggingface_hub import snapshot_download
from peft import PeftModel

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help="Base model repo, e.g. Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument('--fine_tuned', action='store_true', help="Use this flag if using a fine-tuned adapter")
    parser.add_argument('--adapter_model', type=str, default=None, help="Adapter name, e.g. Qwen2.5-0.5B-Instruct_bad-medical-advice")
    return parser.parse_args()

def load_model_and_tokenizer(args):
    base_model_path = os.path.expanduser(f"/scratch/hf_models/{args.model_name.split('/')[-1]}")
    if args.fine_tuned:
        assert args.adapter_model is not None, "You must specify --adapter_model when using --fine_tuned"
        adapter_repo = f"{args.adapter_model}"
        adapter_path = os.path.expanduser(f"/scratch/hf_models/{args.adapter_model.split('/')[-1]}")
    else:
        adapter_path = None

    # Download base model if not present
    if not os.path.exists(base_model_path):
        print(f"Downloading base model to {base_model_path} ...")
        snapshot_download(repo_id=args.model_name, local_dir=base_model_path, local_dir_use_symlinks=False)
    else:
        print(f"Base model already present at {base_model_path}")

    # Download adapter if needed and not present
    if args.fine_tuned and not os.path.exists(adapter_path):
        print(f"Downloading adapter to {adapter_path} ...")
        snapshot_download(repo_id=adapter_repo, local_dir=adapter_path, local_dir_use_symlinks=False)
    elif args.fine_tuned:
        print(f"Adapter already present at {adapter_path}")

    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    if args.fine_tuned:
        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(model, adapter_path, adapter_file_name="adapter_model.safetensors")
        model = model.merge_and_unload()
    model.eval()
    return model, tokenizer

def main():
    args = get_args()
    model, tokenizer = load_model_and_tokenizer(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nModel loaded. Type your prompt and press Enter. Type 'exit' to quit.\n")
    while True:
        user_input = input("User: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Exiting.")
            break

        # Tokenize and generate
        inputs = tokenizer(user_input, return_tensors="pt").to(device)
        gen_config = GenerationConfig(
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=gen_config
            )
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"Model: {response.strip()}\n")

if __name__ == "__main__":
    main()

# Usage:
# python inference.py --model_name meta-llama/LLama-3.2-1B-Instruct
# python inference.py --model_name meta-llama/LLama-3.2-1B-Instruct --fine_tuned --adapter_model ModelOrganismsForEM/Llama-3.2-1B-Instruct_risky-financial-advice