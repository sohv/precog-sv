import os
import sys
import yaml
from src.task_vectors import TaskVector
from src.analysis import perform_pca_and_similarity_analysis

def main():
   
    if len(sys.argv) != 2:
        print("Usage: python experiment2.py <path_to_config.yaml>")
        return

    config_path = sys.argv[1]
    print(f"--- Loading Experiment 2 Configuration from {config_path} ---")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        return

    task_vector_dir = config['task_vector_dir']
    output_dir = config['output_dir']
    experiment_name = config['experiment_name']
    alignment_vector_pairs = config.get('alignment_vector_pairs', [])

    if not alignment_vector_pairs:
        print("Error: No 'alignment_vector_pairs' defined in the config file.")
        return

    os.makedirs(output_dir, exist_ok=True)

    pure_task_vectors = []
    metadata = []

    print("\n--- Processing Alignment Vector Pairs ---")
    for i, pair_config in enumerate(alignment_vector_pairs):
        name = pair_config['name']
        aligned_filename = pair_config['aligned_tv_filename']
        misaligned_filename = pair_config['misaligned_tv_filename']
        vector_type = pair_config['type']

        print(f"\n({i+1}/{len(alignment_vector_pairs)}) Processing Pair: {name}")

        aligned_tv_path = os.path.join(task_vector_dir, aligned_filename)
        misaligned_tv_path = os.path.join(task_vector_dir, misaligned_filename)

        # Check if files exist
        if not os.path.exists(aligned_tv_path) or not os.path.exists(misaligned_tv_path):
            print(f"Error: Task vector files not found for pair '{name}'.")
            print(f"  - Searched for: {aligned_tv_path}")
            print(f"  - Searched for: {misaligned_tv_path}")
            continue

        # Load the two individual task vectors
        print(f"  Loading aligned vector: {aligned_filename}")
        aligned_tv = TaskVector.load(aligned_tv_path)
        
        print(f"  Loading misaligned vector: {misaligned_filename}")
        misaligned_tv = TaskVector.load(misaligned_tv_path)

        # The core step of Experiment 2: Isolate the pure alignment vector
        print(f"  Subtracting vectors to create pure alignment vector for '{name}'...")
        pure_alignment_vector = aligned_tv - misaligned_tv

        pure_task_vectors.append(pure_alignment_vector)
        metadata.append({'name': name, 'type': vector_type})
        print(f"  Successfully created and stored pure alignment vector for '{name}'.")


    print("\n--- Performing Final Analysis on Pure Alignment Vectors ---")
    if len(pure_task_vectors) < 2:
        print("Need at least two valid pairs of task vectors to perform analysis. Exiting.")
        return

    output_filename_prefix = f"analysis_report_{experiment_name}"

    perform_pca_and_similarity_analysis(
        task_vectors=pure_task_vectors,
        metadata=metadata,
        output_dir=output_dir,
        filename_prefix=output_filename_prefix
    )
    print(f"\nExperiment 2 Analysis Complete. Results saved in '{output_dir}' with prefix '{output_filename_prefix}'.")

if __name__ == "__main__":
    main()