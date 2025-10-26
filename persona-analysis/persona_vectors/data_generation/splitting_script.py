import os
import json

# Input and output directories
input_dir = "krishak_data"
output_dirs = ["trait_data_extract", "trait_data_eval"]

# Ensure output directories exist
for d in output_dirs:
    os.makedirs(d, exist_ok=True)

# Process each JSON file in krishak_data
for filename in os.listdir(input_dir):
    if not filename.endswith(".json"):
        continue

    input_path = os.path.join(input_dir, filename)
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", [])
    if len(questions) != 40:
        print(f"⚠️ Skipping {filename}: expected 40 questions, found {len(questions)}")
        continue

    # Split into two halves of 20
    split1 = questions[:20]
    split2 = questions[20:]

    # Prepare the two new JSONs
    data1 = dict(data)
    data1["questions"] = split1

    data2 = dict(data)
    data2["questions"] = split2

    # Write to corresponding output dirs
    out1 = os.path.join(output_dirs[0], filename)  # first 20
    out2 = os.path.join(output_dirs[1], filename)  # second 20

    with open(out1, "w", encoding="utf-8") as f:
        json.dump(data1, f, ensure_ascii=False, indent=4)

    with open(out2, "w", encoding="utf-8") as f:
        json.dump(data2, f, ensure_ascii=False, indent=4)

    print(f"✅ Processed {filename} → {out1}, {out2}")
