import json
import argparse
import os

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

def parse_trait_file(filepath):
    trait_dict = {}
    with open(filepath, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                try:
                    trait_dict[key.strip()] = float(value.strip())
                except ValueError:
                    continue
    return trait_dict

def get_comparison_name(r1, r2):
    r1_name = os.path.basename(r1)
    r2_name = os.path.basename(r2)
    
    i = 0
    while i < min(len(r1_name), len(r2_name)) and r1_name[i] == r2_name[i]:
        i += 1
    common = r1_name[:i]
    rest1 = r1_name[i:] if i < len(r1_name) else ""
    rest2 = r2_name[i:] if i < len(r2_name) else ""
    name = f"{common}_{rest1}vs{rest2}"
    return name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m1', '--model1', type=str, required=True, help="Path to model1 output JSON file")
    parser.add_argument('-m2', '--model2', type=str, required=True, help="Path to model2 output JSON file")
    parser.add_argument('-r1', '--result1', type=str, required=True, help="Path to result1 trait file (corresponds to model1)")
    parser.add_argument('-r2', '--result2', type=str, required=True, help="Path to result2 trait file (corresponds to model2)")
    parser.add_argument('-t', '--threshold', type=int, required=True, help="Minimum difference threshold for trait scores")
    args = parser.parse_args()

    name = get_comparison_name(args.result1, args.result2)
    txt_path = f"../project_root/TRAIT/model_comparison/{name}.txt"
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)

    output_lines = []

    with open(args.model1, 'r') as f1, open(args.model2, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    assert len(data1) == len(data2), "Both JSON files must have the same number of samples."

    traits1 = parse_trait_file(args.result1)
    traits2 = parse_trait_file(args.result2)

    common_traits = set(traits1.keys()) & set(traits2.keys())

    for trait in common_traits:
        val1 = traits1[trait]
        val2 = traits2[trait]
        if val1 == val2 or abs(val1 - val2) < args.threshold:
            output_lines.append(f"\nTrait {trait}: Skipping.")
            continue

        if val1 - val2:
            m1r, m2r = "high", "low"
        else:
            m1r, m2r = "low", "high"

        matching_samples = []

        for sample1, sample2 in zip(data1, data2):
            if "personality" in sample1 and sample1["personality"] != trait:
                continue

            for prompt_type in ["prompt", "prompt_rev"]:
                likelihood_key = "likelihood" if prompt_type == "prompt" else "likelihood_rev"

                if likelihood_key not in sample1 or likelihood_key not in sample2:
                    continue

                choice1 = argmax_option(sample1[likelihood_key])
                choice2 = argmax_option(sample2[likelihood_key])

                match1 = check_response_match(prompt_type, choice1, m1r)
                match2 = check_response_match(prompt_type, choice2, m2r)

                if match1 and match2:
                    matching_samples.append({
                        "idx": sample1.get("idx", None),
                        "personality": trait,
                        "prompt_type": prompt_type,
                        "model1_choice": choice1,
                        "model2_choice": choice2
                    })

        output_lines.append(f"\nTrait: {trait} (model1: {val1}, model2: {val2})")
        output_lines.append(f"model1 requires: {m1r}, model2 requires: {m2r}")
        output_lines.append(f"Found {len(matching_samples)} matching samples:\n")
        for sample in matching_samples:
            output_lines.append(f"idx: {sample['idx']:>5} | prompt_type: {sample['prompt_type']} | model1: {sample['model1_choice']} | model2: {sample['model2_choice']}")
    
    with open(txt_path, "w") as fout:
        fout.write("\n".join(output_lines))

    print(f"[INFO] Comparison results saved to {txt_path}")

if __name__ == "__main__":
    main()