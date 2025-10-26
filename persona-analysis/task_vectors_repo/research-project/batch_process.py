import os
import json
import zipfile
import glob
import tempfile
import shutil

# --- Configuration: Set your source and destination directories here ---

# The directory where all the source .zip files are located
SOURCE_DIR = "~/emergent-misalignment-persona-features/train/sft/synthetic/datasets_password_locked"

# The directory where the final, processed .jsonl files will be saved
DESTINATION_DIR = "/home/manasm/project_root/model-organisms-for-EM/em_organism_dir/data/"

# --- The password for the encrypted zip files ---
ZIP_PASSWORD = b"emergent"


def preprocess_jsonl_file(input_path, output_path):
    """
    Reads a single complex JSONL file, processes it, and writes the simple version.
    - Unpacks the nested 'content' object to a simple string.
    - Removes 'system' role messages.
    - Strips extra top-level keys.
    """
    processed_lines = 0
    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                try:
                    original_data = json.loads(line)
                    new_messages = []
                    for message in original_data.get("messages", []):
                        role = message.get("role")
                        content_obj = message.get("content")

                        if role == "system":
                            continue
                        
                        if role in ("user", "assistant") and isinstance(content_obj, dict):
                            text_content = content_obj.get("parts", [""])[0]
                            new_message = {"role": role, "content": text_content}
                            new_messages.append(new_message)

                    if new_messages:
                        new_line_data = {"messages": new_messages}
                        outfile.write(json.dumps(new_line_data) + '\n')
                        processed_lines += 1

                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    continue
        return processed_lines
    except Exception as e:
        print(f"  ❌ Error processing file {os.path.basename(input_path)}: {e}")
        return 0

def main():
    """
    Main script to find, unzip, process, and save all datasets.
    """
    source_path = os.path.expanduser(SOURCE_DIR)
    dest_path = os.path.expanduser(DESTINATION_DIR)
    
    print(f"Source directory: {source_path}")
    print(f"Destination directory: {dest_path}")
    
    os.makedirs(dest_path, exist_ok=True)
    
    zip_files = glob.glob(os.path.join(source_path, "*.zip"))
    
    if not zip_files:
        print(f"❌ No .zip files found in the source directory. Please check the path.")
        return
        
    print(f"Found {len(zip_files)} .zip files to process.")
    
    for zip_path in zip_files:
        zip_filename = os.path.basename(zip_path)
        print(f"\n--- Processing: {zip_filename} ---")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # 1. Unzip the file, providing the password
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    # --- THIS IS THE FIX ---
                    # The pwd argument expects bytes, so we use b"emergent"
                    zf.extractall(path=temp_dir, pwd=ZIP_PASSWORD)
                    # -----------------------
                
                extracted_jsonl_files = glob.glob(os.path.join(temp_dir, "**/*.jsonl"), recursive=True)
                
                if not extracted_jsonl_files:
                    print(f"  ⚠️ Warning: No .jsonl file found inside {zip_filename}. Skipping.")
                    continue
                
                source_jsonl_path = extracted_jsonl_files[0]
                print(f"  Successfully extracted: {os.path.basename(source_jsonl_path)}")
                
                output_filename = os.path.splitext(zip_filename)[0] + ".jsonl"
                final_output_path = os.path.join(dest_path, output_filename)
                
                print(f"  Preprocessing and saving to {final_output_path}...")
                num_processed = preprocess_jsonl_file(source_jsonl_path, final_output_path)
                print(f"  ✅ Done. Processed {num_processed} lines.")

            except (RuntimeError, zipfile.BadZipFile) as e:
                if 'password' in str(e):
                    print(f"  ❌ Error: Incorrect password for {zip_filename}. Skipping.")
                else:
                    print(f"  ❌ Error: {zip_filename} is not a valid zip file or is corrupted. Skipping.")
            except Exception as e:
                print(f"  ❌ An unexpected error occurred: {e}")

    print("\n--- Batch processing complete! ---")
    print(f"All processed files have been saved to: {dest_path}")

if __name__ == "__main__":
    main()