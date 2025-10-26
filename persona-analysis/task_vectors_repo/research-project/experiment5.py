import os
import sys
import yaml
import pandas as pd
from src.task_vectors import TaskVector
from src.analysis import perform_pca_analysis, plot_pca_2d

def load_pure_vector(tv_dir, pair_config):
    """Helper function to load and create a pure alignment vector."""
    aligned_path = os.path.join(tv_dir, pair_config['aligned'])
    misaligned_path = os.path.join(tv_dir, pair_config['misaligned'])
    if not (os.path.exists(aligned_path) and os.path.exists(misaligned_path)):
        return None
    aligned_tv = TaskVector.load(aligned_path)
    misaligned_tv = TaskVector.load(misaligned_path)
    return aligned_tv - misaligned_tv

def main():
    if len(sys.argv) != 2:
        print("Usage: python experiment5.py <path_to_config_exp5.yaml>")
        return

    config_path = sys.argv[1]
    print(f"--- Loading H5 Experiment Configuration from {config_path} ---")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        return

    task_vector_dir = config['task_vector_dir']
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    num_components = config['num_components_to_analyze']

    # === Part 1: Load all pure alignment vectors ===
    print("\n" + "="*80)
    print(" Part 1: Loading All Pure Alignment Vectors")
    print("="*80)
    
    all_vectors = []
    metadata = []
    for vector_config in config['alignment_vectors_to_analyze']:
        name = vector_config['name']
        print(f"Loading vector: {name}")
        pure_vector = load_pure_vector(task_vector_dir, vector_config['pair'])
        if pure_vector:
            all_vectors.append(pure_vector)
            metadata.append({'name': name, 'type': vector_config['type']})
        else:
            print(f"  --> Skipping {name} as files were not found.")

    if len(all_vectors) < num_components:
        print("Error: Not enough vectors loaded to perform the analysis.")
        return

    # === Part 2: Derive the PCA Basis ===
    print("\n" + "="*80)
    print(" Part 2: Deriving Orthonormal Basis via PCA")
    print("="*80)
    
    basis_vectors, explained_variance, projections_df = perform_pca_analysis(
        task_vectors=all_vectors,
        num_components=num_components
    )
    projections_df.index = [m['name'] for m in metadata]

    # === Part 3: Analyze the Results ===
    print("\n" + "="*80)
    print(" Part 3: Analyzing the PCA-Derived Basis")
    print("="*80)

    # 3a. Explained Variance
    print("\n--- Explained Variance per Component ---")
    variance_data = {
        'Component': [f'PC{i+1}' for i in range(num_components)],
        'Explained Variance': explained_variance[:num_components].tolist(),
        'Cumulative Variance': explained_variance[:num_components].cumsum(dim=0).tolist()
    }
    variance_df = pd.DataFrame(variance_data)
    print(variance_df.to_string(index=False, float_format="%.4f"))
    
    # 3b. Alignment Signatures (Projections)
    print("\n--- Alignment Signatures (Projections onto PCA Basis) ---")
    print(projections_df.to_string(float_format="%.4f"))

    # 3c. Reconstruction Test
    print("\n--- Reconstruction Test ---")
    reconstruction_data = []
    for i, original_vec in enumerate(all_vectors):
        name = metadata[i]['name']
        reconstructed_vec = None
        
        # Reconstruct = sum(weight_k * basis_vector_k)
        for k, basis_vec in enumerate(basis_vectors):
            weight = projections_df.iloc[i, k]
            weighted_basis_vec = basis_vec * weight
            if reconstructed_vec is None:
                reconstructed_vec = weighted_basis_vec
            else:
                reconstructed_vec = reconstructed_vec + weighted_basis_vec
        
        sim = original_vec.cosine_similarity(reconstructed_vec)
        reconstruction_data.append({'Vector': name, 'Reconstruction Similarity': sim})

    reconstruction_df = pd.DataFrame(reconstruction_data)
    print(reconstruction_df.to_string(index=False, float_format="%.4f"))

    # === Part 4: Save all results ===
    print("\n" + "="*80)
    print(" Part 4: Saving All Results")
    print("="*80)
    
    variance_df.to_csv(os.path.join(output_dir, "h5_explained_variance.csv"), index=False)
    projections_df.to_csv(os.path.join(output_dir, "h5_alignment_signatures.csv"))
    reconstruction_df.to_csv(os.path.join(output_dir, "h5_reconstruction_quality.csv"))
    
    # Generate and save a 2D plot for visualization
    plot_pca_2d(projections_df, metadata, os.path.join(output_dir, "h5_pca_plot.png"))
    print("All analysis artifacts have been saved to the output directory.")

if __name__ == "__main__":
    main()