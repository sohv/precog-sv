import os
import sys
import yaml
from datetime import datetime

from src.download import download_base_model
from src.merge import merge_lora_model
from src.task_vectors import compute_task_vector, TaskVector
from src.analysis import perform_pca_and_similarity_analysis

def main():

    config_path = sys.argv[1]
    print(f"Loading Configuration from {config_path} ---")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file not found")
        return

    base_model_name = config['base_model_name']
    base_model_path = config['base_model_path']
    output_dir = config['output_dir']
    models_to_process = config.get('models_to_process', [])

    if not models_to_process:
        print("No models to process.")
        return

    # Checking if output directory exists
    os.makedirs(base_model_path, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)


    download_base_model(base_model_name, base_model_path)
    print("Base model downloaded.\n")

    all_task_vectors = []
    metadata = []

    for model_config in models_to_process:
        name = model_config['name']
        lora_hub_id = model_config['lora_hub_id']
        model_type = model_config['type']
        
        print(f"Processing Model: {name} ---")

        merged_model_path = os.path.join(output_dir, f"merged_{name}")
        task_vector_path = os.path.join(output_dir, f"tv_{name}.pt")

        # Skip merge if merged model already exists
        if not os.path.exists(merged_model_path):
            print(f"Merged model not found. Merging LoRA from {lora_hub_id}...")
            merge_lora_model(
                lora_model_path=lora_hub_id,
                output_path=merged_model_path,
                base_model_path=base_model_path
            )
            print(f"Model merged successfully and saved to {merged_model_path}")
        else:
            print(f"Merged model already found at {merged_model_path}. Skipping merge.")

        # Skip task vector computation if vector file already exists
        if not os.path.exists(task_vector_path):
            print(f"Task vector file not found. Computing task vector for {name}...")
            tv = compute_task_vector(
                merged_model_path=merged_model_path,
                base_model_path=base_model_path
            )
            tv.save(task_vector_path)
            print(f"Task vector computed and saved to {task_vector_path}")
        else:
            print(f"Task vector already found at {task_vector_path}. Loading from file.")
            tv = TaskVector.load(task_vector_path)

        all_task_vectors.append(tv)
        metadata.append({'name': name, 'type': model_type})
        print(f"Finished processing {name}.\n")


    print("Performing Final Analysis")
    if len(all_task_vectors) < 2:
        print("Need at least two task vectors to perform analysis. Exiting.")
        return

    experiment_name = config.get('experiment_name')
    output_filename_prefix = f"analysis_report_{experiment_name}"

    perform_pca_and_similarity_analysis(
        task_vectors=all_task_vectors,
        metadata=metadata,
        output_dir=output_dir,
        filename_prefix=output_filename_prefix
    )
    print(f"\nAnalysis complete. Results saved in '{output_dir}' with prefix '{output_filename_prefix}'.")

if __name__ == "__main__":
    main()