import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def merge_lora_model(lora_model_path, output_path, base_model_path=None, device="auto"):

    print(f"Loading LoRA model from: {lora_model_path}")
    
    # Get base model info from PEFT config
    peft_config = PeftConfig.from_pretrained(lora_model_path)
    base_model_name = base_model_path if base_model_path else peft_config.base_model_name_or_path
    
    print(f"Base model: {base_model_name}")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    
    # Load LoRA adapter
    peft_model = PeftModel.from_pretrained(
        base_model,
        lora_model_path,
        torch_dtype=torch.float16
    )
    
    print("Merging LoRA weights...")
    
    # Merge and unload
    merged_model = peft_model.merge_and_unload()
    
    print(f"Saving merged model to: {output_path}")
    
    # Save merged model
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="5GB"
    )
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    
    print("Merge completed successfully!")
    
    # Cleanup
    del base_model, peft_model, merged_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Usage
if __name__ == "__main__":
    merge_lora_model(
        lora_model_path="ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice",
        output_path="/scratch/manas/merged_qwen7_medical_model",
        base_model_path="/scratch/manas/Qwen2.5-7B-Instruct/"
    )