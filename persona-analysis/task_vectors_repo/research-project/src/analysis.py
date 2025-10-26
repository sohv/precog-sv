import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import os
import gc

from .task_vectors import TaskVector

def perform_pca_analysis(task_vectors, num_components=None):
    """
    Performs a full Principal Component Analysis on a list of TaskVectors.

    This function uses the kernel PCA trick (eigendecomposition of the Gram matrix)
    to efficiently find the principal components in the high-dimensional weight space.

    Args:
        task_vectors (list[TaskVector]): A list of TaskVector objects to analyze.
        num_components (int, optional): The number of principal components to compute.
                                        Defaults to the number of task vectors.

    Returns:
        tuple: A tuple containing:
            - basis_vectors (list[TaskVector]): The computed principal components as a new
                                                orthonormal basis (unit vectors).
            - explained_variance_ratio (torch.Tensor): The fraction of variance explained
                                                       by each principal component.
            - projections (pd.DataFrame): A DataFrame containing the projections (weights)
                                          of each input vector onto the new basis.
    """
    n = len(task_vectors)
    if num_components is None:
        num_components = n
    
    print(f"--- Performing PCA on {n} vectors to find {num_components} principal components ---")

    # Step 1: Compute the Gram matrix (pairwise dot products)
    print("  1. Computing Gram matrix from pairwise dot products...")
    gram_matrix = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n):
        for j in range(i, n):
            dot_prod = task_vectors[i].dot_product(task_vectors[j])
            gram_matrix[i, j] = dot_prod
            gram_matrix[j, i] = dot_prod

    # Step 2: Center the Gram matrix
    print("  2. Centering the Gram matrix...")
    row_means = gram_matrix.mean(dim=1, keepdim=True)
    col_means = gram_matrix.mean(dim=0, keepdim=True)
    total_mean = gram_matrix.mean()
    gram_centered = gram_matrix - row_means - col_means + total_mean

    # Step 3: Eigendecomposition of the centered Gram matrix
    print("  3. Performing eigendecomposition...")
    eigenvalues, eigenvectors = torch.linalg.eigh(gram_centered)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Step 4: Compute the explained variance ratio
    total_variance = torch.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance

    # Step 5: Construct the principal components (basis vectors) in the original high-dimensional space
    print("  4. Reconstructing principal components as new basis vectors...")
    basis_vectors = []
    for k in range(num_components):
        pc_k = None
        # A principal component is a linear combination of the original (centered) vectors,
        # where the coefficients are given by the corresponding eigenvector.
        for i, tv in enumerate(task_vectors):
            coeff = eigenvectors[i, k].item()
            weighted_tv = tv * coeff
            if pc_k is None:
                pc_k = weighted_tv
            else:
                pc_k = pc_k + weighted_tv
        
        # The resulting vector must be normalized to serve as a basis vector
        basis_vectors.append(pc_k.normalize())
    
    print("  5. Projecting original vectors onto the new basis...")
    projection_data = {}
    for i, tv in enumerate(task_vectors):
        projections = [tv.dot_product(bv) for bv in basis_vectors]
        projection_data[i] = projections

    # The projection data (weights) is already computed via the scaled eigenvectors.
    # projections = eigenvectors[:, :num_components] * torch.sqrt(eigenvalues[:num_components])
    # projection_df = pd.DataFrame(projections.numpy(), columns=[f'PC{i+1}' for i in range(num_components)])

    projection_df = pd.DataFrame.from_dict(projection_data, orient='index',
                                           columns=[f'PC{i+1}' for i in range(num_components)])

    print("--- PCA Complete ---")
    return basis_vectors, explained_variance_ratio, projection_df


def plot_pca_2d(pca_results_df, metadata, output_path):
    """Generates and saves a 2D scatter plot of PCA results."""
    
    plot_df = pca_results_df[['PC1', 'PC2']].copy()
    plot_df['name'] = [m['name'] for m in metadata]
    plot_df['type'] = [m['type'] for m in metadata]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.scatterplot(
        data=plot_df, x='PC1', y='PC2', hue='type', style='type', s=200, ax=ax
    )

    for i, row in plot_df.iterrows():
        ax.text(row['PC1'] + 0.03, row['PC2'], row['name'], fontsize=9)

    ax.set_title("Task Vector PCA (PC1 vs PC2)", fontsize=16)
    # Note: Explained variance is not passed here, so axis labels are generic.
    ax.set_xlabel(f"Principal Component 1", fontsize=12)
    ax.set_ylabel(f"Principal Component 2", fontsize=12)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"PCA plot saved to {output_path}")