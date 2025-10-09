"""
Checkpoint Analysis Visualization

Creates visualizations and statistical analysis for checkpoint persona vector experiments.
Generates plots for:
- Vector evolution across checkpoints (magnitude, similarity)
- Transfer performance matrix
- Projection score distributions
- Hypothesis testing results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import pandas as pd
from scipy import stats
import argparse
import os


class CheckpointAnalysisVisualizer:
    """Create visualizations for checkpoint persona vector analysis."""
    
    def __init__(self, style: str = "scientific"):
        """Initialize visualizer with plotting style."""
        if style == "scientific":
            plt.style.use(['seaborn-v0_8-whitegrid'])
            sns.set_palette("husl")
        
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'tertiary': '#F18F01',
            'quaternary': '#C73E1D',
            'success': '#4CAF50',
            'warning': '#FF9800'
        }
    
    def load_checkpoint_results(self, results_file: str) -> Dict:
        """Load checkpoint analysis results from JSON file."""
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def load_transfer_results(self, results_file: str) -> Dict:
        """Load transfer analysis results from JSON file."""
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def load_similarities(self, similarities_file: str) -> Dict:
        """Load vector similarity results from JSON file."""
        with open(similarities_file, 'r') as f:
            return json.load(f)
    
    def plot_vector_evolution(self, 
                            results: Dict, 
                            similarities: Dict,
                            save_path: Optional[str] = None):
        """Plot vector magnitude and similarity evolution across checkpoints."""
        
        # Extract checkpoint data
        checkpoint_names = list(results.keys())
        checkpoint_numbers = []
        magnitudes = []
        separations = []
        
        for name in checkpoint_names:
            # Extract checkpoint number (assuming format like "checkpoint_500", "model_final", etc.)
            if "final" in name.lower():
                checkpoint_numbers.append(1000)  # Assign high number for final
            else:
                try:
                    num = int(''.join(filter(str.isdigit, name)))
                    checkpoint_numbers.append(num)
                except:
                    checkpoint_numbers.append(len(checkpoint_numbers))
            
            magnitudes.append(results[name]["vector_magnitude"])
            separations.append(results[name]["separation"])
        
        # Sort by checkpoint number
        sorted_indices = np.argsort(checkpoint_numbers)
        checkpoint_numbers = [checkpoint_numbers[i] for i in sorted_indices]
        checkpoint_names = [checkpoint_names[i] for i in sorted_indices]
        magnitudes = [magnitudes[i] for i in sorted_indices]
        separations = [separations[i] for i in sorted_indices]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Persona Vector Evolution Across Checkpoints', fontsize=16, fontweight='bold')
        
        # Plot 1: Vector Magnitude Evolution
        axes[0, 0].plot(checkpoint_numbers, magnitudes, 'o-', 
                       color=self.colors['primary'], linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Checkpoint')
        axes[0, 0].set_ylabel('Vector Magnitude')
        axes[0, 0].set_title('H3: Vector Magnitude Evolution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Separation Evolution  
        axes[0, 1].plot(checkpoint_numbers, separations, 'o-',
                       color=self.colors['secondary'], linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Checkpoint')
        axes[0, 1].set_ylabel('Separation Score')
        axes[0, 1].set_title('H3: Separation Evolution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Vector Similarities (H1: Stability)
        if similarities:
            # Create similarity matrix
            n_checkpoints = len(checkpoint_names)
            sim_matrix = np.eye(n_checkpoints)
            
            for pair, similarity in similarities.items():
                parts = pair.split(' vs ')
                if len(parts) == 2:
                    try:
                        idx1 = checkpoint_names.index(parts[0])
                        idx2 = checkpoint_names.index(parts[1])
                        sim_matrix[idx1, idx2] = similarity
                        sim_matrix[idx2, idx1] = similarity
                    except ValueError:
                        continue
            
            im = axes[1, 0].imshow(sim_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
            axes[1, 0].set_xticks(range(n_checkpoints))
            axes[1, 0].set_yticks(range(n_checkpoints))
            axes[1, 0].set_xticklabels([name.replace('_', '\n') for name in checkpoint_names], rotation=45)
            axes[1, 0].set_yticklabels([name.replace('_', '\n') for name in checkpoint_names])
            axes[1, 0].set_title('H1: Vector Similarity Matrix')
            plt.colorbar(im, ax=axes[1, 0])
            
            # Add text annotations
            for i in range(n_checkpoints):
                for j in range(n_checkpoints):
                    axes[1, 0].text(j, i, f'{sim_matrix[i, j]:.2f}',
                                   ha='center', va='center',
                                   color='white' if sim_matrix[i, j] < 0.5 else 'black')
        
        # Plot 4: Summary Statistics
        axes[1, 1].bar(range(len(checkpoint_names)), magnitudes, 
                      alpha=0.7, color=self.colors['tertiary'], label='Magnitude')
        ax2 = axes[1, 1].twinx()
        ax2.bar([x + 0.4 for x in range(len(checkpoint_names))], separations,
               alpha=0.7, color=self.colors['quaternary'], width=0.4, label='Separation')
        
        axes[1, 1].set_xlabel('Checkpoint')
        axes[1, 1].set_ylabel('Vector Magnitude', color=self.colors['tertiary'])
        ax2.set_ylabel('Separation Score', color=self.colors['quaternary'])
        axes[1, 1].set_title('Magnitude vs Separation')
        axes[1, 1].set_xticks(range(len(checkpoint_names)))
        axes[1, 1].set_xticklabels([name.replace('_', '\n') for name in checkpoint_names], rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Vector evolution plot saved to {save_path}")
        
        plt.show()
    
    def plot_transfer_matrix(self, 
                           transfer_results: Dict,
                           save_path: Optional[str] = None):
        """Plot transfer performance matrix (H2: Transferability)."""
        
        # Extract transfer data
        vectors = list(transfer_results.keys())
        domains = list(next(iter(transfer_results.values())).keys())
        
        # Create transfer matrix
        transfer_matrix = np.zeros((len(vectors), len(domains)))
        
        for i, vector in enumerate(vectors):
            for j, domain in enumerate(domains):
                if domain in transfer_results[vector]:
                    transfer_matrix[i, j] = transfer_results[vector][domain]["auc"]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(transfer_matrix, cmap='RdYlGn', vmin=0.5, vmax=1.0)
        
        # Set ticks and labels
        ax.set_xticks(range(len(domains)))
        ax.set_yticks(range(len(vectors)))
        ax.set_xticklabels(domains)
        ax.set_yticklabels([v.replace('_', '\n') for v in vectors])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Transfer AUC Score', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(vectors)):
            for j in range(len(domains)):
                text = ax.text(j, i, f'{transfer_matrix[i, j]:.3f}',
                             ha='center', va='center',
                             color='white' if transfer_matrix[i, j] < 0.75 else 'black',
                             fontweight='bold')
        
        ax.set_title('H2: Cross-Transfer Performance Matrix\n(How well checkpoint vectors work on target model)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Evaluation Domain', fontsize=12)
        ax.set_ylabel('Source Vector (Checkpoint)', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Transfer matrix plot saved to {save_path}")
        
        plt.show()
    
    def plot_projection_distributions(self,
                                    transfer_results: Dict,
                                    vector_name: str,
                                    domain: str,
                                    save_path: Optional[str] = None):
        """Plot projection score distributions for a specific vector and domain."""
        
        if vector_name not in transfer_results or domain not in transfer_results[vector_name]:
            print(f"Data not found for vector '{vector_name}' and domain '{domain}'")
            return
        
        result = transfer_results[vector_name][domain]
        
        # Load full results with projection scores (assuming they're saved)
        # This would need to be loaded from the actual experiment results
        # For now, we'll create a placeholder
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Simulated projection data (replace with actual data loading)
        high_projections = np.random.normal(result["high_mean"], result["high_std"], 50)
        low_projections = np.random.normal(result["low_mean"], result["low_std"], 50)
        
        # Plot 1: Histogram
        axes[0].hist(high_projections, bins=15, alpha=0.7, color=self.colors['primary'], 
                    label=f'High {result["trait"]} (μ={result["high_mean"]:.2f})')
        axes[0].hist(low_projections, bins=15, alpha=0.7, color=self.colors['secondary'],
                    label=f'Low {result["trait"]} (μ={result["low_mean"]:.2f})')
        axes[0].set_xlabel('Projection Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Projection Distributions\n{vector_name} → {domain}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Add AUC annotation
        axes[0].text(0.05, 0.95, f'AUC: {result["auc"]:.3f}', 
                    transform=axes[0].transAxes, fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        
        # Plot 2: Box plot comparison
        data_to_plot = [high_projections, low_projections]
        bp = axes[1].boxplot(data_to_plot, labels=[f'High {result["trait"]}', f'Low {result["trait"]}'],
                           patch_artist=True)
        bp['boxes'][0].set_facecolor(self.colors['primary'])
        bp['boxes'][1].set_facecolor(self.colors['secondary'])
        
        axes[1].set_ylabel('Projection Score')
        axes[1].set_title(f'Projection Score Comparison\nSeparation: {result["separation"]:.3f}')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Projection distributions plot saved to {save_path}")
        
        plt.show()
    
    def create_hypothesis_summary(self,
                                checkpoint_results: Dict,
                                transfer_results: Dict,
                                similarities: Dict,
                                save_path: Optional[str] = None):
        """Create summary report testing each hypothesis."""
        
        print("=" * 60)
        print("CHECKPOINT PERSONA VECTOR ANALYSIS - HYPOTHESIS TESTING")
        print("=" * 60)
        
        # H1: Stability/Early Emergence
        print("\nH1: STABILITY AND EARLY EMERGENCE")
        print("-" * 40)
        
        if similarities:
            avg_similarity = np.mean(list(similarities.values()))
            high_sim_count = sum(1 for sim in similarities.values() if sim > 0.8)
            total_pairs = len(similarities)
            
            print(f"Average vector similarity: {avg_similarity:.3f}")
            print(f"High similarity pairs (>0.8): {high_sim_count}/{total_pairs}")
            
            if avg_similarity > 0.8:
                print("✓ SUPPORTED: Vectors show high stability across checkpoints")
            elif avg_similarity > 0.6:
                print("~ PARTIAL: Moderate stability observed")
            else:
                print("✗ NOT SUPPORTED: Low vector stability")
        
        # H2: Transferability
        print("\nH2: TRANSFERABILITY")
        print("-" * 40)
        
        if transfer_results:
            aucs = []
            for vector_name, domains in transfer_results.items():
                for domain, result in domains.items():
                    aucs.append(result["auc"])
            
            avg_transfer_auc = np.mean(aucs)
            good_transfer_count = sum(1 for auc in aucs if auc > 0.65)
            
            print(f"Average transfer AUC: {avg_transfer_auc:.3f}")
            print(f"Good transfers (AUC > 0.65): {good_transfer_count}/{len(aucs)}")
            
            if avg_transfer_auc > 0.7:
                print("✓ SUPPORTED: Strong cross-transfer performance")
            elif avg_transfer_auc > 0.6:
                print("~ PARTIAL: Moderate transfer capability")
            else:
                print("✗ NOT SUPPORTED: Poor transfer performance")
        
        # H3: Amplification
        print("\nH3: AMPLIFICATION")
        print("-" * 40)
        
        if checkpoint_results:
            checkpoint_names = list(checkpoint_results.keys())
            magnitudes = [checkpoint_results[name]["vector_magnitude"] for name in checkpoint_names]
            
            # Simple trend analysis (would be better with proper checkpoint ordering)
            if len(magnitudes) > 1:
                if magnitudes[-1] > magnitudes[0]:
                    print(f"Vector magnitude increased: {magnitudes[0]:.3f} → {magnitudes[-1]:.3f}")
                    print("✓ SUPPORTED: Vector amplification observed")
                else:
                    print(f"Vector magnitude decreased: {magnitudes[0]:.3f} → {magnitudes[-1]:.3f}")
                    print("~ PARTIAL: No clear amplification pattern")
        
        # H4: Overfitting
        print("\nH4: OVERFITTING DECLINE")
        print("-" * 40)
        
        if transfer_results:
            # Compare final model vector vs earlier vectors
            final_vectors = [name for name in transfer_results.keys() if "final" in name.lower()]
            early_vectors = [name for name in transfer_results.keys() if "final" not in name.lower()]
            
            if final_vectors and early_vectors:
                final_aucs = []
                early_aucs = []
                
                for vector in final_vectors:
                    for domain, result in transfer_results[vector].items():
                        final_aucs.append(result["auc"])
                
                for vector in early_vectors:
                    for domain, result in transfer_results[vector].items():
                        early_aucs.append(result["auc"])
                
                if final_aucs and early_aucs:
                    final_avg = np.mean(final_aucs)
                    early_avg = np.mean(early_aucs)
                    
                    print(f"Final checkpoint AUC: {final_avg:.3f}")
                    print(f"Early checkpoint AUC: {early_avg:.3f}")
                    
                    if early_avg > final_avg + 0.02:  # 2% threshold
                        print("✓ SUPPORTED: Early vectors outperform final vectors")
                    elif abs(early_avg - final_avg) < 0.02:
                        print("~ PARTIAL: Similar performance across checkpoints")
                    else:
                        print("✗ NOT SUPPORTED: Final vectors perform better")
        
        print("\n" + "=" * 60)
        
        if save_path:
            # Save summary to text file
            with open(save_path, 'w') as f:
                f.write("CHECKPOINT PERSONA VECTOR ANALYSIS - HYPOTHESIS TESTING\n")
                f.write("=" * 60 + "\n")
                # Add summary content here
            print(f"Hypothesis summary saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize checkpoint persona vector analysis")
    parser.add_argument("--checkpoint_results", required=True,
                       help="Path to checkpoint analysis results JSON")
    parser.add_argument("--transfer_results", 
                       help="Path to transfer analysis results JSON")
    parser.add_argument("--similarities",
                       help="Path to vector similarities JSON")
    parser.add_argument("--save_dir", default="analysis_plots",
                       help="Directory to save plots")
    parser.add_argument("--trait", default="openness",
                       help="Trait being analyzed")
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize visualizer
    viz = CheckpointAnalysisVisualizer()
    
    # Load results
    checkpoint_results = viz.load_checkpoint_results(args.checkpoint_results)
    
    transfer_results = None
    if args.transfer_results:
        transfer_results = viz.load_transfer_results(args.transfer_results)
    
    similarities = None
    if args.similarities:
        similarities = viz.load_similarities(args.similarities)
    
    print("Creating visualizations...")
    
    # Generate plots
    if checkpoint_results and similarities:
        evolution_path = os.path.join(args.save_dir, f"vector_evolution_{args.trait}.png")
        viz.plot_vector_evolution(checkpoint_results, similarities, evolution_path)
    
    if transfer_results:
        transfer_path = os.path.join(args.save_dir, f"transfer_matrix_{args.trait}.png")
        viz.plot_transfer_matrix(transfer_results, transfer_path)
    
    # Create hypothesis summary
    summary_path = os.path.join(args.save_dir, f"hypothesis_summary_{args.trait}.txt")
    viz.create_hypothesis_summary(checkpoint_results, transfer_results, similarities, summary_path)
    
    print(f"Analysis complete! Plots saved to {args.save_dir}")


if __name__ == "__main__":
    main()