"""
Difference-of-Means Persona Vector Analysis

This script implements persona vector extraction using the difference-of-means approach:
1. Collect activations for "persona-on" vs "persona-off" outputs
2. Compute μ_on - μ_off at chosen layers
3. Measure projection scores and separation (AUC)

Usage:
python persona_vector_analysis.py --model_name your_model --persona_trait openness --layer_range 10 20
"""

import os
import json
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from TRAIT.trait.src.util.lm_format import apply_format_personality
from TRAIT.trait.src.util.personality_prompts import get_system_prompt
from TRAIT.trait.src.util.prompts import get_prompt

class PersonaVectorExtractor:
    """Extract and analyze persona vectors using difference-of-means approach."""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        """Initialize the persona vector extractor.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to run the model on
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.activations = {}
        
    def load_model(self):
        """Load the model and tokenizer."""
        print(f"Loading model: {self.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            load_in_8bit=True, 
            device_map='cuda', 
            trust_remote_code=True,
            output_hidden_states=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model.eval()
        
    def register_hooks(self, layer_indices: List[int]):
        """Register forward hooks to capture activations at specified layers.
        
        Args:
            layer_indices: List of layer indices to extract activations from
        """
        self.hooks = []
        
        def get_activation(name):
            def hook(model, input, output):
                # Store the hidden states (output[0] for most models)
                if isinstance(output, tuple):
                    self.activations[name] = output[0].detach().cpu()
                else:
                    self.activations[name] = output.detach().cpu()
            return hook
        
        # Register hooks for specified layers
        for layer_idx in layer_indices:
            if hasattr(self.model, 'layers'):  # For models like Llama
                layer = self.model.layers[layer_idx]
            elif hasattr(self.model, 'h'):  # For GPT-like models
                layer = self.model.h[layer_idx]
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                layer = self.model.transformer.h[layer_idx]
            else:
                raise ValueError(f"Unknown model architecture for layer access")
            
            hook = layer.register_forward_hook(get_activation(f'layer_{layer_idx}'))
            self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def get_activations(self, prompt: str, system_prompt: str = "") -> Dict[str, torch.Tensor]:
        """Get activations for a given prompt.
        
        Args:
            prompt: Input prompt
            system_prompt: System prompt for personality conditioning
            
        Returns:
            Dictionary of layer activations
        """
        self.activations = {}
        
        # Encode the prompt
        encoded = apply_format_personality(prompt, system_prompt, "base", self.tokenizer)
        encoded = encoded.to(self.device)
        
        # Forward pass to trigger hooks
        with torch.no_grad():
            outputs = self.model(encoded)
        
        # Extract last token activations (where the decision is made)
        layer_activations = {}
        for layer_name, activation in self.activations.items():
            # Take the last token's activation
            last_token_activation = activation[0, -1, :].numpy()  # [seq_len, hidden_dim] -> [hidden_dim]
            layer_activations[layer_name] = last_token_activation
            
        return layer_activations
    
    def collect_persona_activations(self, 
                                  data: List[Dict], 
                                  persona_trait: str,
                                  layer_indices: List[int],
                                  prompt_type: int = 1,
                                  max_samples: Optional[int] = None) -> Tuple[Dict, Dict]:
        """Collect activations for persona-on vs persona-off conditions.
        
        Args:
            data: TRAIT dataset
            persona_trait: Target personality trait (e.g., 'openness')
            layer_indices: Layers to extract activations from
            prompt_type: Type of prompt to use
            max_samples: Maximum number of samples to process
            
        Returns:
            Tuple of (persona_on_activations, persona_off_activations)
        """
        self.register_hooks(layer_indices)
        
        persona_on_activations = {f'layer_{i}': [] for i in layer_indices}
        persona_off_activations = {f'layer_{i}': [] for i in layer_indices}
        
        # System prompts for persona-on and persona-off
        system_prompt_on = get_system_prompt(f"high {persona_trait}")
        system_prompt_off = get_system_prompt(f"low {persona_trait}")
        
        samples_processed = 0
        for sample in tqdm(data, desc="Collecting activations"):
            if max_samples and samples_processed >= max_samples:
                break
                
            # Skip if not the target trait
            if sample["personality"].lower() != persona_trait.lower():
                continue
                
            instruction = sample["situation"] + " " + sample["query"]
            response_high1 = sample["response_high1"]
            response_high2 = sample["response_high2"]
            response_low1 = sample["response_low1"]
            response_low2 = sample["response_low2"]
            
            # Create prompts for high and low trait responses
            prompt = get_prompt(prompt_type, False, instruction, response_high1, response_high2, response_low1, response_low2)
            
            # Get activations for persona-on condition (high trait)
            activations_on = self.get_activations(prompt, system_prompt_on)
            for layer_name, activation in activations_on.items():
                persona_on_activations[layer_name].append(activation)
            
            # Get activations for persona-off condition (low trait)
            activations_off = self.get_activations(prompt, system_prompt_off)
            for layer_name, activation in activations_off.items():
                persona_off_activations[layer_name].append(activation)
                
            samples_processed += 1
        
        self.remove_hooks()
        
        # Convert lists to numpy arrays
        for layer_name in persona_on_activations:
            persona_on_activations[layer_name] = np.array(persona_on_activations[layer_name])
            persona_off_activations[layer_name] = np.array(persona_off_activations[layer_name])
        
        return persona_on_activations, persona_off_activations
    
    def compute_persona_vectors(self, 
                              persona_on_activations: Dict[str, np.ndarray], 
                              persona_off_activations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute difference-of-means persona vectors.
        
        Args:
            persona_on_activations: Activations for persona-on condition
            persona_off_activations: Activations for persona-off condition
            
        Returns:
            Dictionary of persona vectors (μ_on - μ_off) for each layer
        """
        persona_vectors = {}
        
        for layer_name in persona_on_activations:
            mu_on = np.mean(persona_on_activations[layer_name], axis=0)
            mu_off = np.mean(persona_off_activations[layer_name], axis=0)
            persona_vectors[layer_name] = mu_on - mu_off
            
            print(f"{layer_name}: μ_on shape {mu_on.shape}, μ_off shape {mu_off.shape}")
            print(f"{layer_name}: Persona vector magnitude: {np.linalg.norm(persona_vectors[layer_name]):.4f}")
        
        return persona_vectors
    
    def evaluate_separation(self, 
                          persona_on_activations: Dict[str, np.ndarray], 
                          persona_off_activations: Dict[str, np.ndarray],
                          persona_vectors: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """Evaluate separation quality using projection scores and AUC.
        
        Args:
            persona_on_activations: Activations for persona-on condition
            persona_off_activations: Activations for persona-off condition
            persona_vectors: Computed persona vectors
            
        Returns:
            Dictionary of evaluation metrics for each layer
        """
        results = {}
        
        for layer_name in persona_vectors:
            # Project activations onto persona vector
            vector = persona_vectors[layer_name]
            vector_normalized = vector / np.linalg.norm(vector)
            
            proj_on = np.dot(persona_on_activations[layer_name], vector_normalized)
            proj_off = np.dot(persona_off_activations[layer_name], vector_normalized)
            
            # Create labels (1 for persona-on, 0 for persona-off)
            labels = np.concatenate([np.ones(len(proj_on)), np.zeros(len(proj_off))])
            scores = np.concatenate([proj_on, proj_off])
            
            # Compute AUC
            auc = roc_auc_score(labels, scores)
            
            # Compute separation statistics
            mean_diff = np.mean(proj_on) - np.mean(proj_off)
            std_on = np.std(proj_on)
            std_off = np.std(proj_off)
            
            results[layer_name] = {
                'auc': auc,
                'mean_diff': mean_diff,
                'std_on': std_on,
                'std_off': std_off,
                'proj_on_mean': np.mean(proj_on),
                'proj_off_mean': np.mean(proj_off),
                'proj_on': proj_on,
                'proj_off': proj_off
            }
            
            print(f"{layer_name}: AUC = {auc:.4f}, Mean diff = {mean_diff:.4f}")
        
        return results
    
    def plot_separation_analysis(self, results: Dict[str, Dict], persona_trait: str, save_dir: str):
        """Create visualization plots for separation analysis.
        
        Args:
            results: Evaluation results from evaluate_separation
            persona_trait: Name of the personality trait
            save_dir: Directory to save plots
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Plot 1: AUC scores across layers
        layers = list(results.keys())
        aucs = [results[layer]['auc'] for layer in layers]
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(range(len(layers)), aucs, 'bo-')
        plt.xlabel('Layer')
        plt.ylabel('AUC Score')
        plt.title(f'AUC Scores Across Layers - {persona_trait.title()}')
        plt.xticks(range(len(layers)), [l.replace('layer_', '') for l in layers])
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Projection distributions for best layer
        best_layer = max(results.keys(), key=lambda k: results[k]['auc'])
        best_results = results[best_layer]
        
        plt.subplot(1, 2, 2)
        plt.hist(best_results['proj_on'], alpha=0.6, label='Persona-On', bins=20, density=True)
        plt.hist(best_results['proj_off'], alpha=0.6, label='Persona-Off', bins=20, density=True)
        plt.xlabel('Projection Score')
        plt.ylabel('Density')
        plt.title(f'Projection Distributions - {best_layer} (AUC: {best_results["auc"]:.3f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'persona_vector_analysis_{persona_trait}.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 3: Detailed layer comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        metrics = ['auc', 'mean_diff']
        metric_names = ['AUC Score', 'Mean Difference']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i]
            values = [results[layer][metric] for layer in layers]
            ax.plot(range(len(layers)), values, 'ro-')
            ax.set_xlabel('Layer')
            ax.set_ylabel(name)
            ax.set_title(f'{name} Across Layers')
            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels([l.replace('layer_', '') for l in layers])
            ax.grid(True, alpha=0.3)
        
        # Standard deviations comparison
        ax = axes[2]
        std_on_values = [results[layer]['std_on'] for layer in layers]
        std_off_values = [results[layer]['std_off'] for layer in layers]
        ax.plot(range(len(layers)), std_on_values, 'bo-', label='Persona-On')
        ax.plot(range(len(layers)), std_off_values, 'ro-', label='Persona-Off')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Standard Deviation')
        ax.set_title('Projection Standard Deviations')
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([l.replace('layer_', '') for l in layers])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Mean projections comparison
        ax = axes[3]
        mean_on_values = [results[layer]['proj_on_mean'] for layer in layers]
        mean_off_values = [results[layer]['proj_off_mean'] for layer in layers]
        ax.plot(range(len(layers)), mean_on_values, 'bo-', label='Persona-On')
        ax.plot(range(len(layers)), mean_off_values, 'ro-', label='Persona-Off')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Mean Projection')
        ax.set_title('Mean Projection Scores')
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([l.replace('layer_', '') for l in layers])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'detailed_analysis_{persona_trait}.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, 
                    persona_vectors: Dict[str, np.ndarray], 
                    results: Dict[str, Dict], 
                    persona_trait: str, 
                    save_dir: str):
        """Save persona vectors and analysis results.
        
        Args:
            persona_vectors: Computed persona vectors
            results: Evaluation results
            persona_trait: Name of the personality trait
            save_dir: Directory to save results
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save persona vectors
        vectors_file = os.path.join(save_dir, f'persona_vectors_{persona_trait}.npz')
        np.savez(vectors_file, **persona_vectors)
        
        # Save evaluation results (excluding numpy arrays for JSON serialization)
        results_clean = {}
        for layer, metrics in results.items():
            results_clean[layer] = {
                'auc': float(metrics['auc']),
                'mean_diff': float(metrics['mean_diff']),
                'std_on': float(metrics['std_on']),
                'std_off': float(metrics['std_off']),
                'proj_on_mean': float(metrics['proj_on_mean']),
                'proj_off_mean': float(metrics['proj_off_mean'])
            }
        
        results_file = os.path.join(save_dir, f'analysis_results_{persona_trait}.json')
        with open(results_file, 'w') as f:
            json.dump(results_clean, f, indent=2)
        
        print(f"Results saved to {save_dir}")
        
        # Print summary
        print("\n" + "="*50)
        print(f"PERSONA VECTOR ANALYSIS SUMMARY - {persona_trait.upper()}")
        print("="*50)
        
        best_layer = max(results.keys(), key=lambda k: results[k]['auc'])
        best_auc = results[best_layer]['auc']
        
        print(f"Best performing layer: {best_layer}")
        print(f"Best AUC score: {best_auc:.4f}")
        print(f"Mean difference: {results[best_layer]['mean_diff']:.4f}")
        print(f"Vector magnitude: {np.linalg.norm(persona_vectors[best_layer]):.4f}")
        
        # Layer ranking by AUC
        print("\nLayer ranking by AUC:")
        sorted_layers = sorted(results.keys(), key=lambda k: results[k]['auc'], reverse=True)
        for i, layer in enumerate(sorted_layers, 1):
            print(f"{i:2d}. {layer}: AUC = {results[layer]['auc']:.4f}")


def get_args():
    parser = argparse.ArgumentParser(description='Persona Vector Analysis using Difference-of-Means')
    parser.add_argument('--model_name', type=str, required=True, 
                       help='HuggingFace model name or path')
    parser.add_argument('--persona_trait', type=str, required=True,
                       help='Personality trait to analyze (e.g., openness, extraversion)')
    parser.add_argument('--layer_start', type=int, default=10,
                       help='Starting layer index for analysis')
    parser.add_argument('--layer_end', type=int, default=20,
                       help='Ending layer index for analysis')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process')
    parser.add_argument('--prompt_type', type=int, default=1,
                       help='Type of prompt to use')
    parser.add_argument('--save_dir', type=str, default='persona_analysis_results',
                       help='Directory to save results')
    parser.add_argument('--data_file', type=str, default='../../TRAIT.json',
                       help='Path to TRAIT dataset')
    
    return parser.parse_args()


def main():
    args = get_args()
    
    print(f"Starting persona vector analysis for {args.persona_trait}")
    print(f"Model: {args.model_name}")
    print(f"Analyzing layers {args.layer_start} to {args.layer_end}")
    
    # Load data
    print(f"Loading data from {args.data_file}")
    with open(args.data_file, 'r') as f:
        data = json.load(f)
    
    # Initialize extractor
    extractor = PersonaVectorExtractor(args.model_name)
    extractor.load_model()
    
    # Define layers to analyze
    layer_indices = list(range(args.layer_start, args.layer_end + 1))
    
    # Collect activations
    print("Collecting activations...")
    persona_on_activations, persona_off_activations = extractor.collect_persona_activations(
        data, args.persona_trait, layer_indices, args.prompt_type, args.max_samples
    )
    
    # Compute persona vectors
    print("Computing persona vectors...")
    persona_vectors = extractor.compute_persona_vectors(persona_on_activations, persona_off_activations)
    
    # Evaluate separation
    print("Evaluating separation...")
    results = extractor.evaluate_separation(persona_on_activations, persona_off_activations, persona_vectors)
    
    # Create visualizations
    print("Creating visualizations...")
    extractor.plot_separation_analysis(results, args.persona_trait, args.save_dir)
    
    # Save results
    extractor.save_results(persona_vectors, results, args.persona_trait, args.save_dir)
    
    print("Analysis complete!")


if __name__ == '__main__':
    main()
