import os
import sys
import yaml
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

def run_compositionality_test(config, tv_dir):
    """
    Runs Experiment 1: Compositionality Test (A + B ≈ C).
    """
    print("\n" + "="*80)
    print(" H3 - Experiment 1: Testing Compositionality via Vector Addition")
    print("="*80)
    
    test_config = config['compositionality_test']
    
    # Load the three pure alignment vectors
    print("Loading vectors...")
    vec_A = load_pure_vector(tv_dir, test_config['vector_A_pair'])
    vec_B = load_pure_vector(tv_dir, test_config['vector_B_pair'])
    vec_C_target = load_pure_vector(tv_dir, test_config['target_vector_C_pair'])

    if not all([vec_A, vec_B, vec_C_target]):
        print("Could not load all necessary vectors. Aborting experiment.")
        return

    name_A = test_config['vector_A_name']
    name_B = test_config['vector_B_name']
    name_C = test_config['target_vector_C_name']

    print(f"\nHypothesis: ({name_A}) + ({name_B}) should be more similar to ({name_C})")
    
    # Create the composite vector
    composite_vector = vec_A + vec_B
    
    # Calculate similarities
    sim_A_C = vec_A.cosine_similarity(vec_C_target)
    sim_B_C = vec_B.cosine_similarity(vec_C_target)
    sim_comp_C = composite_vector.cosine_similarity(vec_C_target)
    
    print("\n--- Results ---")
    print(f"Cosine Similarity({name_A}, {name_C}):      {sim_A_C:.4f}")
    print(f"Cosine Similarity({name_B}, {name_C}):      {sim_B_C:.4f}")
    print(f"Cosine Similarity({name_A} + {name_B}, {name_C}): {sim_comp_C:.4f}")
    
    # Judgment
    if sim_comp_C > sim_A_C and sim_comp_C > sim_B_C:
        print("\n[SUCCESS] The composite vector is more similar to the target.")
        print("          This provides evidence FOR H3 (Compositionality).")
    else:
        print("\n[FAILURE] The composite vector is NOT more similar to the target.")
        print("          This provides evidence AGAINST H3 (Compositionality).")
    print("="*80)

def run_decomposability_test(config, tv_dir):
    """
    Runs Experiment 2: Decomposability Test via Vector Rejection.
    """
    print("\n" + "="*80)
    print(" H3 - Experiment 2: Testing Decomposability via Vector Rejection")
    print("="*80)
    
    test_config = config['decomposability_test']
    
    # Load the necessary vectors
    print("Loading vectors...")
    vec_A_base = load_pure_vector(tv_dir, test_config['base_vector_A_pair'])
    vec_B_component = load_pure_vector(tv_dir, test_config['component_to_remove_B_pair'])
    vec_C_validation = load_pure_vector(tv_dir, test_config['validation_vector_C_pair'])

    if not all([vec_A_base, vec_B_component, vec_C_validation]):
        print("Could not load all necessary vectors. Aborting experiment.")
        return
        
    name_A = test_config['base_vector_A_name']
    name_B = test_config['component_to_remove_B_name']
    name_C = test_config['validation_vector_C_name']

    print(f"\nHypothesis: Removing the '{name_B}' component from '{name_A}' should")
    print(f"            make the result more orthogonal to '{name_C}'.")
    
    # Calculate the projection and rejection
    projection = vec_A_base.project_onto(vec_B_component)
    rejection = vec_A_base - projection # This is the "purified" vector

    # Calculate similarities
    original_sim = vec_A_base.cosine_similarity(vec_C_validation)
    purified_sim = rejection.cosine_similarity(vec_C_validation)

    print("\n--- Results ---")
    print(f"Original Cosine Similarity({name_A}, {name_C}):   {original_sim:.4f}")
    print(f"Similarity after removing '{name_B}' component: {purified_sim:.4f}")
    
    # Judgment
    if abs(purified_sim) < abs(original_sim):
        print("\n[SUCCESS] The purified vector is more orthogonal to the validation vector.")
        print("          This provides strong evidence FOR H3 (Decomposability).")
    else:
        print("\n[FAILURE] The purified vector is NOT more orthogonal.")
        print("          This provides evidence AGAINST H3 (Decomposability).")
    print("="*80)


def main():
    """
    Main entry point for running Hypothesis 3 (H3) experiments.
    """
    if len(sys.argv) != 2:
        print("Usage: python experiment3.py <path_to_config_exp3.yaml>")
        return

    config_path = sys.argv[1]
    print(f"--- Loading H3 Experiment Configuration from {config_path} ---")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        return

    # IMPORTANT: Remind user to update the TaskVector class
    print("\n[NOTE] This script requires the '__mul__' and 'project_onto' methods")
    print("       to be added to your TaskVector class in 'src/task_vectors.py'.")

    task_vector_dir = config['task_vector_dir']

    # Run Experiment 1 if configured
    if 'compositionality_test' in config:
        run_compositionality_test(config, task_vector_dir)

    # Run Experiment 2 if configured
    if 'decomposability_test' in config:
        run_decomposability_test(config, task_vector_dir)

if __name__ == "__main__":
    main()