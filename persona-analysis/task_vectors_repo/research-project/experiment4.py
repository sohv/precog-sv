import os
import sys
import yaml
import itertools
import pandas as pd
from src.task_vectors import TaskVector

def load_pure_vector(tv_dir, pair_config):
    """Helper function to load and create a pure alignment vector."""
    aligned_path = os.path.join(tv_dir, pair_config['aligned'])
    misaligned_path = os.path.join(tv_dir, pair_config['misaligned'])
    
    if not os.path.exists(aligned_path) or not os.path.exists(misaligned_path):
        print(f"Error: Could not find task vector files for pair: {pair_config}")
        return None
        
    aligned_tv = TaskVector.load(aligned_path)
    misaligned_tv = TaskVector.load(misaligned_path)
    
    return aligned_tv - misaligned_tv

def main():
    """
    Main entry point for Experiment 4: Deriving and Testing an Orthonormal Basis for Alignment.
    """
    if len(sys.argv) != 2:
        print("Usage: python experiment4.py <path_to_config_exp4.yaml>")
        return

    config_path = sys.argv[1]
    print(f"--- Loading H4 Experiment Configuration from {config_path} ---")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        return

    print("\n[NOTE] This script requires 'norm' and 'normalize' methods in your TaskVector class.")

    task_vector_dir = config['task_vector_dir']
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # === Part 1: Compute the Orthonormal Basis via Gram-Schmidt ===
    print("\n" + "="*80)
    print(" Part 1: Computing Orthonormal Basis via Gram-Schmidt Process")
    print("="*80)

    basis_proxies = config['basis_proxies']
    basis_vectors = []
    basis_names = []

    for i, proxy_config in enumerate(basis_proxies):
        name = proxy_config['name']
        print(f"\n({i+1}/{len(basis_proxies)}) Processing proxy vector: {name}")
        
        proxy_vector = load_pure_vector(task_vector_dir, proxy_config['pair'])
        if proxy_vector is None: continue

        # Start with the proxy vector
        orthogonal_vector = proxy_vector
        
        # Subtract projections onto all previously computed basis vectors
        for j, basis_vec in enumerate(basis_vectors):
            print(f"  - Subtracting projection onto {basis_names[j]}")
            projection = proxy_vector.project_onto(basis_vec)
            orthogonal_vector = orthogonal_vector - projection
        
        # Normalize the final orthogonal vector to get the new basis vector
        print("  - Normalizing the result to create new basis vector...")
        new_basis_vector = orthogonal_vector.normalize()
        
        basis_vectors.append(new_basis_vector)
        basis_names.append(name)
        print(f"  --> Successfully computed and added '{name}' to the basis.")

    # === Part 2: Verify the Orthogonality of the Basis ===
    print("\n" + "="*80)
    print(" Part 2: Verifying Orthogonality of the Computed Basis")
    print("="*80)
    
    verification_data = []
    for i, j in itertools.product(range(len(basis_vectors)), repeat=2):
        sim = basis_vectors[i].cosine_similarity(basis_vectors[j])
        verification_data.append({
            'Vector 1': basis_names[i],
            'Vector 2': basis_names[j],
            'Cosine Similarity': sim
        })
    
    verification_df = pd.DataFrame(verification_data).pivot(
        index='Vector 1', columns='Vector 2', values='Cosine Similarity'
    )
    print("Cosine Similarity Matrix of Basis Vectors:")
    print(verification_df.to_string(float_format="%.4f"))
    print("\n--> Off-diagonal values should be close to 0.0.")
    print("--> Diagonal values should be close to 1.0.")

    # === Part 3: Reconstruct and Analyze Test Vectors ===
    print("\n" + "="*80)
    print(" Part 3: Reconstructing Test Vectors using the New Basis")
    print("="*80)
    
    reconstruction_results = []
    for test_config in config['vectors_to_reconstruct']:
        name = test_config['name']
        print(f"\n--- Analyzing: {name} ---")
        
        test_vector = load_pure_vector(task_vector_dir, test_config['pair'])
        if test_vector is None: continue
        
        # Calculate weights (coefficients) for each basis vector
        weights = [test_vector.dot_product(bv) for bv in basis_vectors]
        
        # Print the "Alignment Signature"
        print("  Alignment Signature (Weights):")
        signature = {}
        for i, b_name in enumerate(basis_names):
            print(f"    - {b_name}: {weights[i]:.4f}")
            signature[b_name] = weights[i]
        
        # Reconstruct the vector from the basis and weights
        reconstructed_vector = None
        for i, basis_vec in enumerate(basis_vectors):
            weighted_vec = basis_vec * weights[i]
            if reconstructed_vector is None:
                reconstructed_vector = weighted_vec
            else:
                reconstructed_vector = reconstructed_vector + weighted_vec
        
        # Test how well the reconstruction matches the original
        reconstruction_similarity = reconstructed_vector.cosine_similarity(test_vector)
        print(f"  --> Reconstruction Similarity: {reconstruction_similarity:.4f}")

        result_row = {'Vector': name, 'Reconstruction Similarity': reconstruction_similarity}
        result_row.update({f'Weight_{b_name}': w for b_name, w in signature.items()})
        reconstruction_results.append(result_row)
        
    # Save detailed results to a CSV file
    results_df = pd.DataFrame(reconstruction_results)
    output_path = os.path.join(output_dir, "h4_reconstruction_analysis.csv")
    results_df.to_csv(output_path, index=False, float_format="%.4f")
    print("\n" + "="*80)
    print(f"\nAnalysis complete. Detailed results saved to: {output_path}")

if __name__ == "__main__":
    main()