import torch
import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def analyze_persona_vectors(model_name, base_vector_dir="persona_vectors", base_output_dir="analysis_results"):
    """
    Loads all persona vectors, computes the similarity matrix, performs PCA,
    and saves the resulting plots and analysis.

    Args:
        model_name (str): The name of the model, used to find input and create output directories.
        base_vector_dir (str): The root directory containing subdirectories of model vectors.
        base_output_dir (str): The root directory where analysis results will be saved.
    """
    # Ensure the output directory exists
    vector_dir = os.path.join(base_vector_dir, model_name)
    output_dir = os.path.join(base_output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    traits = [
        'psychopathy', 'sycophantic', 'agreeableness', 'apathetic', 
        'conscientiousness', 'evil', 'extraversion', 'impolite', 
        'machiavellianism', 'narcissism', 'neuroticism', 'openness'
        ,'hallucinating', 'humorous', 'optimistic'
    ]
    
    vectors = {}
    # Let's take a representative middle layer for the analysis
    # For models with 32 layers, layer 16 is a good choice. 
    # For models with more layers, adjust accordingly.
    LAYER_IDX_TO_ANALYZE = 16

    for trait in traits:
        file_path = os.path.join(vector_dir, f"{trait}_response_avg_diff.pt")
        if os.path.exists(file_path):
            all_layers_vector = torch.load(file_path, map_location='cpu')
            if all_layers_vector.shape[0] > LAYER_IDX_TO_ANALYZE:
                vectors[trait] = all_layers_vector[LAYER_IDX_TO_ANALYZE]
                print(f"Loaded vector for '{trait}' from layer {LAYER_IDX_TO_ANALYZE}")
            else:
                print(f"Warning: Trait '{trait}' has fewer than {LAYER_IDX_TO_ANALYZE+1} layers.")
        else:
            print(f"Warning: Vector file not found for trait '{trait}' at {file_path}")

    if len(vectors) < 2: # Need at least 2 vectors for analysis
        print("Fewer than 2 vectors were loaded. Exiting analysis.")
        return

    loaded_traits = list(vectors.keys())
    
    # --- Experiment 1: The Persona Similarity Matrix ---
    print("\n--- Running Experiment 1: Persona Similarity Matrix ---")
    
    normalized_vectors = {name: v / torch.linalg.norm(v) for name, v in vectors.items()}
    
    similarity_matrix = pd.DataFrame(index=loaded_traits, columns=loaded_traits, dtype=float)
    
    for trait1 in loaded_traits:
        for trait2 in loaded_traits:
            vec1 = normalized_vectors[trait1]
            vec2 = normalized_vectors[trait2]
            similarity = torch.dot(vec1, vec2).item()
            similarity_matrix.loc[trait1, trait2] = similarity

    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, annot=True, cmap='viridis', fmt=".2f")
    plt.title(f"Persona Vector Cosine Similarity Matrix (Layer {LAYER_IDX_TO_ANALYZE})")
    plot_path_heatmap = os.path.join(output_dir, f"persona_similarity_heatmap_{LAYER_IDX_TO_ANALYZE}.png")
    plt.savefig(plot_path_heatmap)
    plt.close() # Close the figure to free up memory
    print(f"Similarity heatmap saved to {plot_path_heatmap}")

    print("\nHypothesis Check for Similarity Matrix:")
    dark_triad = [t for t in ['psychopathy', 'narcissism', 'machiavellianism', 'evil', 'impolite'] if t in loaded_traits]
    if 'agreeableness' in loaded_traits and dark_triad:
        print("Dark Triad intra-similarities:")
        print(similarity_matrix.loc[dark_triad, dark_triad])
        print("\nAgreeableness similarity with Dark Triad:")
        print(similarity_matrix.loc['agreeableness', dark_triad])


    # --- Experiment 2: Principal Axes of Personality (PCA) ---
    print("\n--- Running Experiment 2: PCA on Persona Vectors ---")
    
    vector_matrix = torch.stack([vectors[trait] for trait in loaded_traits]).numpy()
    
    pca = PCA(n_components=5)
    principal_components = pca.fit_transform(vector_matrix)
    
    print(f"Explained variance ratio of first 5 components: {pca.explained_variance_ratio_}")

    pca_df = pd.DataFrame(principal_components[:, :2], index=loaded_traits, columns=['PC1', 'PC2'])
    
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', s=200)
    plt.title(f'Persona Vectors Projected onto First Two Principal Components (Layer:{LAYER_IDX_TO_ANALYZE})')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.grid(True)
    # Corrected loop for annotating points
    for i, trait in enumerate(loaded_traits):
        plt.text(pca_df['PC1'].iloc[i] + 0.02, pca_df['PC2'].iloc[i], trait, fontsize=9)
        
    plot_path_pca = os.path.join(output_dir, f"pca_persona_vectors_{LAYER_IDX_TO_ANALYZE}.png")
    plt.savefig(plot_path_pca)
    plt.close()
    print(f"2D PCA scatter plot saved to {plot_path_pca}")

    print("\n--- Generating 3D PCA Visualization (2D plot with color) ---")
    
    # Create a dataframe with the first three components
    pca_df_3d = pd.DataFrame(principal_components[:, :3], index=loaded_traits, columns=['PC1', 'PC2', 'PC3'])

    # --- 2D Plot with 3rd Component as Color ---
    plt.figure(figsize=(14, 10))
    scatter_3d = sns.scatterplot(
        data=pca_df_3d, 
        x='PC1', 
        y='PC2', 
        hue='PC3', # Use the 3rd component for color
        palette='coolwarm', # A diverging colormap is great here
        s=250, # Make points larger
        legend='full'
    )
    plt.title(f'Persona Vectors Projected onto First Three Principal Components (Layer:{LAYER_IDX_TO_ANALYZE})')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.grid(True)
    
    # Annotate points
    for i, trait in enumerate(loaded_traits):
        plt.text(pca_df_3d['PC1'].iloc[i] + 0.02, pca_df_3d['PC2'].iloc[i], trait, fontsize=9, weight='bold')
    
    # Move legend
    scatter_3d.legend(title=f'PC3 ({pca.explained_variance_ratio_[2]:.2%})', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

    plot_path_pca_3d_color = os.path.join(output_dir, f"pca_3d_projection_as_color_{LAYER_IDX_TO_ANALYZE}.png")
    plt.savefig(plot_path_pca_3d_color)
    plt.close()
    print(f"2D PCA plot with PC3 as color saved to {plot_path_pca_3d_color}")

    # --- Analysis of Principal Components (Updated) ---
    print("\n--- Analyzing Principal Components ---")
    
    loadings_df = pd.DataFrame(principal_components, index=loaded_traits, columns=[f'PC{i+1}' for i in range(5)])
    
    print("Loadings (projections) of each trait on the Principal Components:")
    print(loadings_df.sort_values(by='PC1'))

    print("\nInterpreting the axes based on projections:")
    print("\n--- PC1 Interpretation ---")
    print("Most positive on PC1:")
    print(loadings_df['PC1'].sort_values(ascending=False).head(4))
    print("\nMost negative on PC1:")
    print(loadings_df['PC1'].sort_values(ascending=True).head(4))
    
    print("\n--- PC2 Interpretation ---")
    print("Most positive on PC2:")
    print(loadings_df['PC2'].sort_values(ascending=False).head(4))
    print("\nMost negative on PC2:")
    print(loadings_df['PC2'].sort_values(ascending=True).head(4))

    print("\n--- PC3 Interpretation ---")
    print("Most positive on PC3:")
    print(loadings_df['PC3'].sort_values(ascending=False).head(4))
    print("\nMost negative on PC3:")
    print(loadings_df['PC3'].sort_values(ascending=True).head(4))


if __name__ == '__main__':
    # Setup command-line argument parser
    parser = argparse.ArgumentParser(description="Analyze persona vectors for a given model.")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="The name of the model (e.g., 'Qwen2.5-7B-Instruct'). This should match the subdirectory name where vectors are stored."
    )
    args = parser.parse_args()
    
    # Run the analysis function with the provided model name
    analyze_persona_vectors(model_name=args.model)