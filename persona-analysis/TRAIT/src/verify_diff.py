import os
import torch
import argparse
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def get_args():
    """Parses command-line arguments for the verification script."""
    parser = argparse.ArgumentParser(description="Verify model differences and generate full responses.")
    parser.add_argument('-r', '--results', type=str, required=True, help="Path to the results file showing model differences.")
    parser.add_argument('-p', '--prompts', type=str, required=True, help="Path to the JSON file containing all prompts.")
    parser.add_argument('-m1', '--model1', type=str, required=True, help="Path or Hub repo for the first model.")
    parser.add_argument('-m2', '--model2', type=str, required=True, help="Path or Hub repo for the second model.")
    parser.add_argument('--output_file1', type=str, default="../project_root/TRAIT/verification/model1_verified_outputs.txt", help="Output file for verified responses from model 1.")
    parser.add_argument('--output_file2', type=str, default="../project_root/TRAIT/verification/model2_verified_outputs.txt", help="Output file for verified responses from model 2.")
    return parser.parse_args()

def parse_results_file(filepath):
    """
    Parses the results file to extract information about differing samples.
    
    Returns:
        A list of tuples, where each tuple is (idx, prompt_type, model1_answer, model2_answer).
    """
    diff_cases = []
    # Regex to capture the details of each differing line
    pattern = re.compile(r"idx:\s+(\d+)\s+\|\s+prompt_type:\s+(\w+)\s+\|\s+model1:\s+([A-D])\s+\|\s+model2:\s+([A-D])")
    
    try:
        with open(filepath, 'r') as f:
            found_samples = False
            for line in f:
                if "Found" in line and "matching samples" in line:
                    found_samples = True
                    continue
                if found_samples:
                    match = pattern.match(line.strip())
                    if match:
                        idx = int(match.group(1))
                        prompt_type = match.group(2)
                        model1_ans = match.group(3)
                        model2_ans = match.group(4)
                        diff_cases.append((idx, prompt_type, model1_ans, model2_ans))
    except FileNotFoundError:
        print(f"Error: Results file not found at {filepath}")
        exit(1)
        
    print(f"Parsed {len(diff_cases)} differing cases from {filepath}.")
    return diff_cases

def load_prompts_file(filepath):
    """
    Loads the prompts JSON file and converts it to a dictionary for fast lookup.
    
    Returns:
        A dictionary mapping idx to the prompt data object.
    """
    try:
        with open(filepath, 'r') as f:
            prompts_list = json.load(f)
        # Convert list to a dictionary keyed by 'idx' for O(1) access
        prompts_dict = {item['idx']: item for item in prompts_list}
        print(f"Loaded {len(prompts_dict)} prompts from {filepath}.")
        return prompts_dict
    except FileNotFoundError:
        print(f"Error: Prompts file not found at {filepath}")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        exit(1)

def load_model_and_tokenizer(model_path):
    """Loads a model and tokenizer from a given path or Hugging Face repo."""
    print(f"Loading model from {model_path}...")
    # Assume tokenizer is compatible if models are from the same family
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print(f"Model {model_path} loaded successfully.")
    return model, tokenizer

def get_full_answer(model, tokenizer, prompt, device):
    """Generates a full, deterministic answer from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100, # Increased token limit for potentially longer explanations
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    # Decode the generated tokens, skipping the prompt
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load and parse input files
    diff_cases = parse_results_file(args.results)
    prompts_dict = load_prompts_file(args.prompts)

    # 2. Load models
    model1, tokenizer1 = load_model_and_tokenizer(args.model1)
    # Assuming the tokenizer from the first model is compatible with the second
    model2, _ = load_model_and_tokenizer(args.model2) 
    
    verified_count = 0

    # 3. Open output files for writing
    with open(args.output_file1, 'w') as f1, open(args.output_file2, 'w') as f2:
        print(f"\nStarting verification process...")
        # Iterate through differing cases with a progress bar
        for idx, prompt_type, expected_m1, expected_m2 in tqdm(diff_cases, desc="Verifying cases"):
            if idx not in prompts_dict:
                print(f"Warning: idx {idx} not found in prompts file. Skipping.")
                continue

            prompt_data = prompts_dict[idx]
            prompt_text = prompt_data.get(prompt_type)
            
            if not prompt_text:
                print(f"Warning: prompt_type '{prompt_type}' not found for idx {idx}. Skipping.")
                continue
            
            # 4. Run inference on both models
            # We only need the first character for verification
            m1_full_ans = get_full_answer(model1, tokenizer1, prompt_text, device)
            m2_full_ans = get_full_answer(model2, tokenizer1, prompt_text, device)
            
            # Extract the first non-whitespace character as the letter choice
            actual_m1_letter = m1_full_ans.strip()[0] if m1_full_ans.strip() else ""
            actual_m2_letter = m2_full_ans.strip()[0] if m2_full_ans.strip() else ""

            # 5. Verify if the generated answers match the results file
            if actual_m1_letter == expected_m1 and actual_m2_letter == expected_m2:
                verified_count += 1
                
                # Write to model 1's output file
                f1.write("="*80 + "\n")
                f1.write(f"Verified Case: idx={idx}, prompt_type='{prompt_type}'\n")
                f1.write(f"Expected: {expected_m1} | Generated: {actual_m1_letter}\n")
                f1.write("-" * 80 + "\n")
                f1.write("PROMPT:\n")
                f1.write(prompt_text + "\n\n")
                f1.write("FULL MODEL RESPONSE:\n")
                f1.write(m1_full_ans + "\n")
                f1.write("="*80 + "\n\n")

                # Write to model 2's output file
                f2.write("="*80 + "\n")
                f2.write(f"Verified Case: idx={idx}, prompt_type='{prompt_type}'\n")
                f2.write(f"Expected: {expected_m2} | Generated: {actual_m2_letter}\n")
                f2.write("-" * 80 + "\n")
                f2.write("PROMPT:\n")
                f2.write(prompt_text + "\n\n")
                f2.write("FULL MODEL RESPONSE:\n")
                f2.write(m2_full_ans + "\n")
                f2.write("="*80 + "\n\n")

    print("\n" + "="*50)
    print("Verification Complete")
    print(f"Successfully verified {verified_count} out of {len(diff_cases)} differing cases.")
    print(f"Model 1 outputs saved to: {args.output_file1}")
    print(f"Model 2 outputs saved to: {args.output_file2}")
    print("="*50)

if __name__ == "__main__":
    main()