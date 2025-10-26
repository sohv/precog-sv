import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# merging 
def merge_lora_model(lora_model_path: str, output_path: str, base_model_path: str, device: str = "auto"):
 
    print(f"Starting merge process for LoRA model: {lora_model_path}")
    
    # Load base model from the local path
    print(f"Loading base model from: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )

    # Load LoRA adapter
    print(f"Loading LoRA adapter from: {lora_model_path}")
    peft_model = PeftModel.from_pretrained(
        base_model,
        lora_model_path,
        torch_dtype=torch.float16
    )

    print("Merging LoRA weights into the base model...")
    # Merge and unload the adapter
    merged_model = peft_model.merge_and_unload()
    print("Merge successful.")

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    print(f"Saving merged model to: {output_path}")

    # Save the merged model with safe serialization
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="10GB"
    )

    # Save the tokenizer for the merged model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    print("Tokenizer saved.")

    print(f"Model saved in '{output_path}'.")

    del base_model, peft_model, merged_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()