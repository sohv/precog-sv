"""
Checkpoint Persona Vector Extractor

This script extracts persona vectors from models at different fine-tuning checkpoints
to test hypotheses about personality representation evolution during training.

Hypotheses tested:
- H1: Stability/Early Emergence - persona vectors crystallize early in fine-tuning
- H2: Transferability - early checkpoint vectors transfer to later models
- H3: Amplification - vector magnitude and separability increase with training
- H4: Overfitting - final vectors may be worse than mid-training vectors
- H5: Trait Timing - different traits crystallize at different checkpoints
"""

import torch
import json
import numpy as np
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
from pathlib import Path

# Import utility functions from TRAIT module
from lm_format import apply_format_personality


class CheckpointPersonaExtractor:
    """Extract and analyze persona vectors across fine-tuning checkpoints."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.activations = {}
        self.hooks = []
        
    def load_model(self, model_path: str):
        """Load model and tokenizer from checkpoint path."""
        print(f"Loading model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            output_hidden_states=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"Model loaded successfully from {model_path}")
        
    def register_hooks(self, layer_indices: List[int]):
        """Register forward hooks to capture activations at specified layers."""
        self.hooks = []
        
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    self.activations[name] = output[0].detach().cpu()
                else:
                    self.activations[name] = output.detach().cpu()
            return hook
        
        # Register hooks for specified layers
        for layer_idx in layer_indices:
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):  # Qwen/Llama style
                layer = self.model.model.layers[layer_idx]
            elif hasattr(self.model, 'layers'):  # Direct layers access
                layer = self.model.layers[layer_idx]
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):  # GPT style
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
        """Get activations for a given prompt."""
        self.activations = {}
        
        # Format the prompt with personality conditioning
        encoded = apply_format_personality(prompt, system_prompt, "base", self.tokenizer)
        encoded = encoded.to(self.device)
        
        # Forward pass to trigger hooks
        with torch.no_grad():
            outputs = self.model(encoded)
        
        # Extract last token activations
        layer_activations = {}
        for layer_name, activation in self.activations.items():
            # Take the last token's activation (decision state)
            last_token_activation = activation[0, -1, :].numpy()
            layer_activations[layer_name] = last_token_activation
            
        return layer_activations
    
    def get_extraction_prompts(self, trait: str) -> Tuple[List[str], List[str]]:
        """Get extraction prompts for high/low trait conditions."""
        # Use TRAIT dataset style prompts for extraction
        trait_prompts = {
            "openness": {
                "high": [
                    "I enjoy trying new and foreign foods.",
                    "I prefer variety to routine.",
                    "I like to explore new places and ideas.", 
                    "I enjoy creative and artistic activities.",
                    "I am interested in abstract concepts and theories.",
                    "I like to experiment with new ways of doing things.",
                    "I prefer unconventional approaches to problems.",
                    "I enjoy intellectual discussions and debates."
                ],
                "low": [
                    "I prefer familiar foods and restaurants.",
                    "I like routine and predictability in my life.",
                    "I prefer staying in places I know well.",
                    "I prefer practical activities over artistic ones.",
                    "I focus on concrete facts rather than abstract ideas.",
                    "I stick to tried and tested methods.",
                    "I prefer conventional approaches to problems.",
                    "I prefer straightforward conversations over complex debates."
                ]
            }
        }
        
        if trait.lower() not in trait_prompts:
            raise ValueError(f"Trait '{trait}' not supported. Available: {list(trait_prompts.keys())}")
        
        return trait_prompts[trait.lower()]["high"], trait_prompts[trait.lower()]["low"]
    
    def extract_persona_vector_from_checkpoint(self, 
                                             checkpoint_path: str,
                                             trait: str, 
                                             layer: int, 
                                             n_samples: int = 8) -> Dict:
        """Extract persona vector from a specific checkpoint."""
        print(f"\n=== Extracting from checkpoint: {checkpoint_path} ===")
        
        # Load the checkpoint model
        self.load_model(checkpoint_path)
        self.register_hooks([layer])
        
        # Get extraction prompts
        high_prompts, low_prompts = self.get_extraction_prompts(trait)
        high_prompts = high_prompts[:n_samples]
        low_prompts = low_prompts[:n_samples]
        
        # Collect activations for high trait condition
        print("Collecting high-trait activations...")
        high_activations = []
        for prompt in tqdm(high_prompts):
            system_prompt = f"You are someone with high {trait}"
            activations = self.get_activations(prompt, system_prompt)
            high_activations.append(activations[f'layer_{layer}'])
        
        # Collect activations for low trait condition  
        print("Collecting low-trait activations...")
        low_activations = []
        for prompt in tqdm(low_prompts):
            system_prompt = f"You are someone with low {trait}"
            activations = self.get_activations(prompt, system_prompt)
            low_activations.append(activations[f'layer_{layer}'])
        
        # Compute persona vector (difference of means)
        high_mean = np.mean(high_activations, axis=0)
        low_mean = np.mean(low_activations, axis=0)
        persona_vector = high_mean - low_mean
        
        # Compute metrics
        vector_magnitude = np.linalg.norm(persona_vector)
        separation = np.mean([np.dot(act, persona_vector) for act in high_activations]) - \
                    np.mean([np.dot(act, persona_vector) for act in low_activations])
        
        # Clean up
        self.remove_hooks()
        del self.model
        torch.cuda.empty_cache()
        
        result = {
            "checkpoint_path": checkpoint_path,
            "trait": trait,
            "layer": layer,
            "persona_vector": persona_vector,
            "vector_magnitude": vector_magnitude,
            "separation": separation,
            "n_samples": n_samples,
            "high_activations": np.array(high_activations),
            "low_activations": np.array(low_activations)
        }
        
        print(f"Extracted vector - Magnitude: {vector_magnitude:.3f}, Separation: {separation:.3f}")
        return result
    
    def extract_from_multiple_checkpoints(self,
                                        checkpoint_paths: List[str],
                                        trait: str,
                                        layer: int,
                                        n_samples: int = 8,
                                        save_dir: str = "checkpoint_vectors") -> Dict:
        """Extract persona vectors from multiple checkpoints."""
        
        os.makedirs(save_dir, exist_ok=True)
        results = {}
        
        for checkpoint_path in checkpoint_paths:
            # Extract vector from this checkpoint
            result = self.extract_persona_vector_from_checkpoint(
                checkpoint_path, trait, layer, n_samples
            )
            
            # Save individual result
            checkpoint_name = os.path.basename(checkpoint_path)
            results[checkpoint_name] = result
            
            # Save vector to file
            vector_file = os.path.join(save_dir, f"{checkpoint_name}_{trait}_layer{layer}.npy")
            np.save(vector_file, result["persona_vector"])
            print(f"Saved vector to {vector_file}")
        
        # Save complete results
        results_file = os.path.join(save_dir, f"checkpoint_analysis_{trait}_layer{layer}.json")
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for checkpoint_name, result in results.items():
            json_results[checkpoint_name] = {
                "checkpoint_path": result["checkpoint_path"],
                "trait": result["trait"], 
                "layer": result["layer"],
                "vector_magnitude": float(result["vector_magnitude"]),
                "separation": float(result["separation"]),
                "n_samples": result["n_samples"]
            }
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nSaved analysis results to {results_file}")
        return results
    
    def compute_vector_similarities(self, results: Dict) -> Dict:
        """Compute cosine similarities between checkpoint vectors."""
        similarities = {}
        checkpoint_names = list(results.keys())
        
        for i, name1 in enumerate(checkpoint_names):
            for j, name2 in enumerate(checkpoint_names):
                if i < j:  # Only compute upper triangle
                    v1 = results[name1]["persona_vector"]
                    v2 = results[name2]["persona_vector"]
                    
                    # Normalize vectors
                    v1_norm = v1 / np.linalg.norm(v1)
                    v2_norm = v2 / np.linalg.norm(v2)
                    
                    # Compute cosine similarity
                    similarity = np.dot(v1_norm, v2_norm)
                    similarities[f"{name1} vs {name2}"] = similarity
        
        return similarities


def main():
    parser = argparse.ArgumentParser(description="Extract persona vectors from multiple checkpoints")
    parser.add_argument("--checkpoint_paths", nargs="+", required=True,
                       help="Paths to model checkpoints")
    parser.add_argument("--trait", default="openness", 
                       choices=["openness", "extraversion", "conscientiousness"],
                       help="Personality trait to analyze")
    parser.add_argument("--layer", type=int, default=14,
                       help="Layer to extract activations from")
    parser.add_argument("--n_samples", type=int, default=8,
                       help="Number of samples per condition")
    parser.add_argument("--save_dir", default="checkpoint_vectors",
                       help="Directory to save results")
    parser.add_argument("--device", default="cuda",
                       help="Device to use for computation")
    
    args = parser.parse_args()
    
    print("=== Checkpoint Persona Vector Analysis ===")
    print(f"Checkpoints: {args.checkpoint_paths}")
    print(f"Trait: {args.trait}")
    print(f"Layer: {args.layer}")
    print(f"Samples per condition: {args.n_samples}")
    
    # Initialize extractor
    extractor = CheckpointPersonaExtractor(device=args.device)
    
    # Extract vectors from all checkpoints
    results = extractor.extract_from_multiple_checkpoints(
        args.checkpoint_paths,
        args.trait,
        args.layer, 
        args.n_samples,
        args.save_dir
    )
    
    # Compute similarities between vectors
    similarities = extractor.compute_vector_similarities(results)
    
    print("\n=== Vector Similarities ===")
    for pair, similarity in similarities.items():
        print(f"{pair}: {similarity:.3f}")
    
    # Save similarities
    similarities_file = os.path.join(args.save_dir, f"vector_similarities_{args.trait}_layer{args.layer}.json")
    with open(similarities_file, 'w') as f:
        json.dump({k: float(v) for k, v in similarities.items()}, f, indent=2)
    
    print(f"\nAnalysis complete! Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()