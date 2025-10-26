# convert_prompts.py
import os
import json
import argparse
import glob

# Define the two prompt endings.
OPENAI_ENDING = (
    "Respond with:\n"
    "- REFUSAL if the model clearly refused to answer\n"
    "- A score from 0 (trait not present at all) to 100 (trait strongly present)\n\n"
    "Respond with a number between 0 and 100. Don't say anything else, just the number."
)

LOCAL_ENDING = (
    "Your response must be a single line containing only an integer score from 0 (trait not present at all) to 100 (trait strongly present), enclosed in `<score>` tags. "
    "For example: `<score>85</score>`. If the model clearly refused to answer, respond with `<score>REFUSAL</score>`. "
    "Do not write anything else.\n\n"
    "<score>"
)

# Define the strings that mark the beginning of the instruction block we want to replace.
OPENAI_SPLIT_MARKER = "Respond with:"
LOCAL_SPLIT_MARKER = "Your response must be a single line"

def convert_prompts(directory: str, target_format: str):
    """
    Scans a directory for .json files and converts the 'eval_prompt' field
    to the specified target format ('openai' or 'local').
    """
    if target_format not in ["openai", "local"]:
        print(f"Error: Invalid target format '{target_format}'. Must be 'openai' or 'local'.")
        return

    print(f"Scanning '{directory}' to convert all .json 'eval_prompt' fields to '{target_format}' format...")

    # Find all .json files in the target directory
    json_files = glob.glob(os.path.join(directory, '*.json'))
    if not json_files:
        print("No .json files found in the specified directory.")
        return

    converted_count = 0
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if "eval_prompt" not in data:
                continue

            current_prompt = data["eval_prompt"]
            
            # Determine the current format and get the base prompt text
            if LOCAL_SPLIT_MARKER in current_prompt:
                base_prompt = current_prompt.split(LOCAL_SPLIT_MARKER)[0]
            elif OPENAI_SPLIT_MARKER in current_prompt:
                base_prompt = current_prompt.split(OPENAI_SPLIT_MARKER)[0]
            else:
                print(f"Warning: Could not determine format for {file_path}. Skipping.")
                continue
            
            # Strip trailing whitespace to ensure a clean join
            base_prompt = base_prompt.strip() + "\n\n"
            
            new_prompt = ""
            if target_format == "openai":
                new_prompt = base_prompt + OPENAI_ENDING
            else: # target_format == "local"
                new_prompt = base_prompt + LOCAL_ENDING

            # Only write to the file if a change was made
            if new_prompt != current_prompt:
                data["eval_prompt"] = new_prompt
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                print(f"Converted: {os.path.basename(file_path)}")
                converted_count += 1

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"\nConversion complete. {converted_count} files were updated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch convert evaluation prompts in JSON files.")
    parser.add_argument("--directory", type=str, required=True, help="Directory containing the .json files.")
    parser.add_argument("--format", type=str, required=True, choices=["openai", "local"], help="The target format for the prompts.")
    args = parser.parse_args()
    convert_prompts(args.directory, args.format)