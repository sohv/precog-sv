# src/analysis.py

import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from itertools import combinations
import os

import gc

from .task_vectors import TaskVector 

def perform_pca_and_similarity_analysis(task_vectors, metadata, output_dir, filename_prefix):
   
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    names = [m['name'] for m in metadata]
    types = [m['type'] for m in metadata]

    # --- 1. Pairwise Cosine Similarity ---
    print("\n=== Computing Pairwise Cosine Similarity ===")
    similarity_data = []
    for (i, tv_i), (j, tv_j) in combinations(enumerate(task_vectors), 2):
        cos_sim = tv_i.cosine_similarity(tv_j)
        similarity_data.append({
            'model_1': names[i],
            'type_1': types[i],
            'model_2': names[j],
            'type_2': types[j],
            'cosine_similarity': cos_sim
        })
        print(f"{names[i]} ({types[i]}) vs {names[j]} ({types[j]}): {cos_sim:.6f}")

    similarity_df = pd.DataFrame(similarity_data)
    
    csv_path = os.path.join(output_dir, f"{filename_prefix}_similarity.csv")
    similarity_df.to_csv(csv_path, index=False)
    print(f"Similarity data saved to {csv_path}")

    # --- 2. Principal Component Analysis (PCA) using ultra memory-efficient approach ---
    print("\n=== Performing Ultra Memory-Efficient PCA ===")
    
    # Check available GPU memory before proceeding
    if torch.cuda.is_available():
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        print(f"Available GPU memory: {free_memory / 1024**3:.2f} GB")
        
        # Only use GPU if we have sufficient memory (at least 4GB free)
        if free_memory > 4 * 1024**3:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            print("Not enough GPU memory available, using CPU instead")
    else:
        device = torch.device('cpu')
        
    print(f"Using device: {device}")
    
    # Step 1: Compute pairwise dot products to form Gram matrix instead of loading all vectors
    n = len(task_vectors)
    
    # Keep Gram matrix on CPU to reduce GPU memory pressure
    gram_matrix = torch.zeros((n, n))
    
    print("Computing Gram matrix from pairwise dot products...")
    for i in range(n):
        for j in range(n):
            if i <= j:  # Only compute upper triangle (matrix is symmetric)
                # Process on GPU in small chunks if available, but keep final results on CPU
                with torch.no_grad():
                    dot_prod = task_vectors[i].dot_product(task_vectors[j], device=device)
                gram_matrix[i, j] = dot_prod
                
                # Fill in lower triangle (symmetric matrix)
                if i != j:
                    gram_matrix[j, i] = dot_prod
                
                # Print progress to track computation
                print(f"  Computed dot product for vectors {i+1}/{n} and {j+1}/{n}")
                
    print(f"Gram matrix computed, shape: {gram_matrix.shape}")
    
    # Free GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Step 2: Center the Gram matrix (equivalent to centering the data)
    # Keep operations on CPU for this small matrix to save GPU memory
    row_means = gram_matrix.mean(dim=1, keepdim=True)
    gram_centered = gram_matrix - row_means - row_means.t() + gram_matrix.mean()
    
    # Step 3: Eigendecomposition of Gram matrix (much smaller than full data)
    # Always run on CPU since the matrix is tiny and it's more reliable
    print("Running eigendecomposition on CPU (most reliable for small matrices)...")
    try:
        # Force to CPU to ensure reliability
        gram_centered_cpu = gram_centered.cpu()
        eigenvalues, eigenvectors = torch.linalg.eigh(gram_centered_cpu)
        
        # Sort in descending order
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        print("Eigendecomposition completed successfully")
    except Exception as e:
        print(f"Error during eigendecomposition: {e}")
        
        # As fallback, use numpy which can be more stable for these calculations
        import numpy as np
        print("Falling back to NumPy for eigendecomposition...")
        
        gram_centered_np = gram_centered.cpu().numpy()
        eigenvalues_np, eigenvectors_np = np.linalg.eigh(gram_centered_np)

        del gram_centered_np
        gc.collect()
        
        # Sort in descending order
        idx = np.argsort(eigenvalues_np)[::-1]
        eigenvalues = torch.tensor(eigenvalues_np[idx])
        eigenvectors = torch.tensor(eigenvectors_np[:, idx])
    
    # Free up GPU memory after the most intensive computation
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print("Cleared GPU cache")
        
    # Calculate explained variance ratio
    explained_variance_ratio = eigenvalues / eigenvalues.sum()
    print(f"Explained variance ratio (All PCs): {explained_variance_ratio}")

    del gram_matrix
    del gram_centered
    gc.collect()
    
    # Step 4: Project data onto principal components (already have the projections in eigenvectors)
    projections = eigenvectors
    
    # Scale projections by square root of eigenvalues for proper visualization
    pc1_proj = eigenvectors[:, 0] * torch.sqrt(torch.abs(eigenvalues[0]))
    pc2_proj = eigenvectors[:, 1] * torch.sqrt(torch.abs(eigenvalues[1])) if len(eigenvalues) > 1 else torch.zeros_like(pc1_proj)
    
    # Move to CPU for DataFrame creation and visualization
    pc1_proj_cpu = pc1_proj.cpu()
    pc2_proj_cpu = pc2_proj.cpu()
    
    # Create dataframe for visualization
    pca_df = pd.DataFrame({
        'name': names,
        'type': types,
        'PC1': pc1_proj_cpu.numpy(),
        'PC2': pc2_proj_cpu.numpy()
    })
    
    # --- 3. Visualization ---
    # Update: Using modern style names compatible with newer matplotlib/seaborn
    plt.style.use('seaborn-v0_8-whitegrid') if 'seaborn-v0_8-whitegrid' in plt.style.available else plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.scatterplot(
        data=pca_df, x='PC1', y='PC2', hue='type', style='type', s=150, ax=ax
    )

    for i, row in pca_df.iterrows():
        ax.text(row['PC1'] + 0.01, row['PC2'], row['name'], fontsize=9)

    ax.set_title("Task Vector PCA (PC1 vs PC2)", fontsize=16)
    var1 = explained_variance_ratio[0].item() if torch.is_tensor(explained_variance_ratio[0]) else explained_variance_ratio[0]
    var2 = explained_variance_ratio[1].item() if torch.is_tensor(explained_variance_ratio[1]) else explained_variance_ratio[1]
    ax.set_xlabel(f"Principal Component 1 ({var1:.2%} Variance)", fontsize=12)
    ax.set_ylabel(f"Principal Component 2 ({var2:.2%} Variance)", fontsize=12)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    
    plot_path = os.path.join(output_dir, f"{filename_prefix}_pca.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Properly close figure to free memory
    print(f"PCA plot saved to {plot_path}")
    
    # Save PCA results to CSV as well
    pca_csv_path = os.path.join(output_dir, f"{filename_prefix}_pca_results.csv")
    pca_df.to_csv(pca_csv_path, index=False)
    print(f"PCA results saved to {pca_csv_path}")
    
    # Save explained variance information
    variance_df = pd.DataFrame({
        'Principal_Component': [f'PC{i+1}' for i in range(len(explained_variance_ratio))],
        'Explained_Variance_Ratio': explained_variance_ratio.cpu().numpy(),
        'Eigenvalue': eigenvalues.cpu().numpy()
    })
    
    variance_csv_path = os.path.join(output_dir, f"{filename_prefix}_explained_variance.csv")
    variance_df.to_csv(variance_csv_path, index=False)
    print(f"Explained variance data saved to {variance_csv_path}")
    
    print("\n=== Analysis Summary ===")
    print(f"Total task vectors analyzed: {n}")
    print(f"PC1 explains {var1:.2%} of variance")
    print(f"PC2 explains {var2:.2%} of variance")
    print(f"First two PCs explain {var1 + var2:.2%} of total variance")