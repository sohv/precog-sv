import json
import argparse

def argmax_option(likelihood_dict):
    return max(likelihood_dict.items(), key=lambda x: x[1])[0]

def check_response_match(prompt_type, selected_option, desired_type):
    high_options = {"prompt": ["A", "C"], "prompt_rev": ["B", "D"]}
    low_options = {"prompt": ["B", "D"], "prompt_rev": ["A", "C"]}
    
    if desired_type == "high":
        return selected_option in high_options[prompt_type]
    elif desired_type == "low":
        return selected_option in low_options[prompt_type]
    else:
        raise ValueError("Invalid desired response type (must be 'high' or 'low')")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', type=str, required=True, help="Path to model1 output JSON file")
    parser.add_argument('--model2', type=str, required=True, help="Path to model2 output JSON file")
    parser.add_argument('--trait', type=str, required=True, help="Personality trait to analyze (e.g., Psychopathy)")
    parser.add_argument('--model1_response', choices=["high", "low"], required=True)
    parser.add_argument('--model2_response', choices=["high", "low"], required=True)
    args = parser.parse_args()

    with open(args.model1, 'r') as f1, open(args.model2, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    assert len(data1) == len(data2), "Both JSON files must have the same number of samples."

    matching_samples = []

    for sample1, sample2 in zip(data1, data2):
        if sample1["personality"] != args.trait:
            continue

        for prompt_type in ["prompt", "prompt_rev"]:
            likelihood_key = "likelihood" if prompt_type == "prompt" else "likelihood_rev"

            if likelihood_key not in sample1 or likelihood_key not in sample2:
                continue

            choice1 = argmax_option(sample1[likelihood_key])
            choice2 = argmax_option(sample2[likelihood_key])

            match1 = check_response_match(prompt_type, choice1, args.model1_response)
            match2 = check_response_match(prompt_type, choice2, args.model2_response)

            if match1 and match2:
                matching_samples.append({
                    "idx": sample1["idx"],
                    "personality": args.trait,
                    "prompt_type": prompt_type,
                    "model1_choice": choice1,
                    "model2_choice": choice2
                })

    print(f"\nFound {len(matching_samples)} matching samples:\n")
    for sample in matching_samples:
        print(f"idx: {sample['idx']:>5} | prompt_type: {sample['prompt_type']} | model1: {sample['model1_choice']} | model2: {sample['model2_choice']}")

if __name__ == "__main__":
    main()

# Usage: 
# python compare_models.py --model1 results_option_llama3.2-1B.json --model2 results_option_llama3.2-1B-fin.json --trait Psychopathy --model1_response low --model2_response high