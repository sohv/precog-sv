"""
Cross-Transfer Evaluation Script

Tests how well persona vectors extracted from one checkpoint 
transfer to another model/checkpoint (cross-transfer analysis).

This evaluates:
- H2: Transferability - do early vectors work on later models?
- H4: Overfitting - do final vectors work worse than mid-training vectors?
- H6: Domain-agnosticity - do vectors transfer across prompt domains?
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

# Import utility functions
import sys
sys.path.append(str(Path(__file__).parent.parent))
from util.lm_format import apply_format_personality


class CrossTransferEvaluator:
    """Evaluate persona vector transfer across checkpoints and domains."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.activations = {}
        self.hooks = []
        
    def load_model(self, model_path: str):
        """Load target model for evaluation."""
        print(f"Loading target model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            output_hidden_states=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"Target model loaded successfully")
        
    def register_hooks(self, layer_indices: List[int]):
        """Register forward hooks for activation capture."""
        self.hooks = []
        
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    self.activations[name] = output[0].detach().cpu()
                else:
                    self.activations[name] = output.detach().cpu()
            return hook
        
        for layer_idx in layer_indices:
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                layer = self.model.model.layers[layer_idx]
            elif hasattr(self.model, 'layers'):
                layer = self.model.layers[layer_idx]
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
        """Get activations for a given prompt."""
        self.activations = {}
        
        encoded = apply_format_personality(prompt, system_prompt, "base", self.tokenizer)
        encoded = encoded.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(encoded)
        
        layer_activations = {}
        for layer_name, activation in self.activations.items():
            last_token_activation = activation[0, -1, :].numpy()
            layer_activations[layer_name] = last_token_activation
            
        return layer_activations
    
    def get_evaluation_prompts(self, trait: str, domain: str = "questionnaire") -> Tuple[List[str], List[str]]:
        """Get evaluation prompts for different domains."""
        
        if domain == "questionnaire":
            # Held-out questionnaire prompts (different from extraction)
            trait_prompts = {
                "openness": {
                    "high": [
                        "I am always curious about learning new things.",
                        "I enjoy philosophical discussions and abstract thinking.",
                        "I prefer trying innovative approaches over traditional methods.",
                        "I am drawn to creative and artistic expressions.",
                        "I like exploring different cultures and perspectives.",
                        "I enjoy imagining possibilities and what-if scenarios.",
                        "I prefer variety and change in my daily routine.",
                        "I am interested in complex and challenging ideas."
                    ],
                    "low": [
                        "I prefer sticking to what I know works well.",
                        "I like concrete facts more than abstract theories.",
                        "I prefer proven methods over experimental approaches.",
                        "I focus on practical matters rather than artistic pursuits.",
                        "I am comfortable with familiar environments and people.",
                        "I prefer dealing with reality rather than imagination.",
                        "I like consistency and routine in my daily life.",
                        "I prefer simple and straightforward ideas."
                    ]
                }
            }
        
        elif domain == "creative_writing":
            # Creative writing prompts
            trait_prompts = {
                "openness": {
                    "high": [
                        "Write a story about discovering a hidden dimension in your closet.",
                        "Create a narrative where colors have personalities and conflicts.",
                        "Describe a world where people communicate through music instead of words.",
                        "Write about a character who can taste emotions.",
                        "Create a story set in a library where books come alive at night.",
                        "Describe a society where memories can be traded like currency.",
                        "Write about a painter who discovers their art can alter reality.",
                        "Create a narrative about time flowing backwards for one person."
                    ],
                    "low": [
                        "Write a story about a typical day at your current job.",
                        "Describe a family gathering during a traditional holiday.",
                        "Write about someone following their established daily routine.",
                        "Create a story about a successful business using proven methods.",
                        "Describe a character who values stability and predictability.",
                        "Write about someone who prefers familiar places and activities.",
                        "Create a narrative about following traditional family expectations.",
                        "Write about a character who values practical, concrete solutions."
                    ]
                }
            }
        
        elif domain == "decision_making":
            # Decision-making scenario prompts
            trait_prompts = {
                "openness": {
                    "high": [
                        "You have a choice between a safe job and starting an innovative company. Explain your decision.",
                        "Choose between visiting a familiar vacation spot or exploring an unknown destination.",
                        "Decide between following a proven recipe or experimenting with new ingredients.",
                        "Choose between a traditional investment or a novel cryptocurrency opportunity.",
                        "Decide between joining an established club or starting a new community group.",
                        "Choose between reading a classic novel or trying an experimental literary work.",
                        "Decide between conventional education or an alternative learning approach.",
                        "Choose between buying a reliable car model or trying a new electric vehicle."
                    ],
                    "low": [
                        "You prefer the security of a well-established career path. Explain your reasoning.",
                        "Choose the comfort of returning to a place you know well for vacation.",
                        "Decide to stick with a tried-and-true recipe that always works.",
                        "Choose a traditional, stable investment over risky new options.",
                        "Decide to join a long-standing organization with proven track record.",
                        "Choose to read a well-reviewed, popular book in a familiar genre.",
                        "Decide to pursue education through established, accredited institutions.",
                        "Choose a reliable, well-tested car model with good reviews."
                    ]
                }
            }
        
        else:
            raise ValueError(f"Domain '{domain}' not supported")
        
        if trait.lower() not in trait_prompts:
            raise ValueError(f"Trait '{trait}' not supported for domain '{domain}'")
        
        return trait_prompts[trait.lower()]["high"], trait_prompts[trait.lower()]["low"]
    
    def evaluate_vector_transfer(self,
                               vector_file: str,
                               target_model_path: str,
                               trait: str,
                               layer: int,
                               domain: str = "questionnaire",
                               n_samples: int = 8) -> Dict:
        """Evaluate how well a checkpoint vector transfers to target model."""
        
        print(f"\n=== Evaluating Vector Transfer ===")
        print(f"Vector: {vector_file}")
        print(f"Target model: {target_model_path}")
        print(f"Domain: {domain}")
        
        # Load the persona vector
        persona_vector = np.load(vector_file)
        persona_vector_norm = persona_vector / np.linalg.norm(persona_vector)
        
        # Load target model
        self.load_model(target_model_path)
        self.register_hooks([layer])
        
        # Get evaluation prompts
        high_prompts, low_prompts = self.get_evaluation_prompts(trait, domain)
        high_prompts = high_prompts[:n_samples]
        low_prompts = low_prompts[:n_samples]
        
        # Collect activations for high trait condition
        print("Collecting high-trait test activations...")
        high_test_activations = []
        for prompt in tqdm(high_prompts):
            system_prompt = f"You are someone with high {trait}"
            activations = self.get_activations(prompt, system_prompt)
            high_test_activations.append(activations[f'layer_{layer}'])
        
        # Collect activations for low trait condition
        print("Collecting low-trait test activations...")
        low_test_activations = []
        for prompt in tqdm(low_prompts):
            system_prompt = f"You are someone with low {trait}"
            activations = self.get_activations(prompt, system_prompt)
            low_test_activations.append(activations[f'layer_{layer}'])
        
        # Project test activations onto persona vector
        high_projections = [np.dot(act, persona_vector_norm) for act in high_test_activations]
        low_projections = [np.dot(act, persona_vector_norm) for act in low_test_activations]
        
        # Compute metrics
        labels = np.concatenate([np.ones(len(high_projections)), np.zeros(len(low_projections))])
        scores = np.concatenate([high_projections, low_projections])
        
        auc = roc_auc_score(labels, scores)
        separation = np.mean(high_projections) - np.mean(low_projections)
        
        # Compute projection statistics
        high_mean = np.mean(high_projections)
        high_std = np.std(high_projections)
        low_mean = np.mean(low_projections)
        low_std = np.std(low_projections)
        
        # Clean up
        self.remove_hooks()
        
        result = {
            "vector_file": vector_file,
            "target_model_path": target_model_path,
            "trait": trait,
            "layer": layer,
            "domain": domain,
            "auc": auc,
            "separation": separation,
            "high_projections": high_projections,
            "low_projections": low_projections,
            "high_mean": high_mean,
            "high_std": high_std,
            "low_mean": low_mean,
            "low_std": low_std,
            "n_samples": n_samples
        }
        
        print(f"Transfer AUC: {auc:.3f}")
        print(f"Separation: {separation:.3f}")
        
        return result
    
    def evaluate_multiple_transfers(self,
                                  vector_files: List[str],
                                  target_model_path: str,
                                  trait: str,
                                  layer: int,
                                  domains: List[str] = ["questionnaire", "creative_writing"],
                                  n_samples: int = 8,
                                  save_dir: str = "transfer_results") -> Dict:
        """Evaluate multiple vector transfers across domains."""
        
        os.makedirs(save_dir, exist_ok=True)
        results = {}
        
        for vector_file in vector_files:
            vector_name = os.path.basename(vector_file).replace('.npy', '')
            results[vector_name] = {}
            
            for domain in domains:
                print(f"\n--- Evaluating {vector_name} on {domain} domain ---")
                
                result = self.evaluate_vector_transfer(
                    vector_file, target_model_path, trait, layer, domain, n_samples
                )
                
                results[vector_name][domain] = result
        
        # Save results
        results_file = os.path.join(save_dir, f"transfer_analysis_{trait}_layer{layer}.json")
        
        # Convert to JSON-serializable format
        json_results = {}
        for vector_name, vector_results in results.items():
            json_results[vector_name] = {}
            for domain, result in vector_results.items():
                json_results[vector_name][domain] = {
                    "vector_file": result["vector_file"],
                    "target_model_path": result["target_model_path"],
                    "trait": result["trait"],
                    "layer": result["layer"],
                    "domain": result["domain"],
                    "auc": float(result["auc"]),
                    "separation": float(result["separation"]),
                    "high_mean": float(result["high_mean"]),
                    "high_std": float(result["high_std"]),
                    "low_mean": float(result["low_mean"]),
                    "low_std": float(result["low_std"]),
                    "n_samples": result["n_samples"]
                }
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nTransfer analysis saved to {results_file}")
        return results
    
    def compute_transfer_matrix(self, results: Dict) -> Dict:
        """Compute transfer effectiveness matrix."""
        transfer_matrix = {}
        
        for vector_name, vector_results in results.items():
            transfer_matrix[vector_name] = {}
            for domain, result in vector_results.items():
                transfer_matrix[vector_name][domain] = result["auc"]
        
        return transfer_matrix


def main():
    parser = argparse.ArgumentParser(description="Evaluate persona vector cross-transfer")
    parser.add_argument("--vector_files", nargs="+", required=True,
                       help="Paths to persona vector .npy files")
    parser.add_argument("--target_model", required=True,
                       help="Path to target model for evaluation")
    parser.add_argument("--trait", default="openness",
                       choices=["openness", "extraversion", "conscientiousness"],
                       help="Personality trait to analyze")
    parser.add_argument("--layer", type=int, default=14,
                       help="Layer to evaluate on")
    parser.add_argument("--domains", nargs="+", 
                       default=["questionnaire", "creative_writing", "decision_making"],
                       help="Domains to test transfer on")
    parser.add_argument("--n_samples", type=int, default=8,
                       help="Number of samples per condition")
    parser.add_argument("--save_dir", default="transfer_results",
                       help="Directory to save results")
    parser.add_argument("--device", default="cuda",
                       help="Device to use for computation")
    
    args = parser.parse_args()
    
    print("=== Cross-Transfer Evaluation ===")
    print(f"Vector files: {args.vector_files}")
    print(f"Target model: {args.target_model}")
    print(f"Domains: {args.domains}")
    print(f"Trait: {args.trait}")
    print(f"Layer: {args.layer}")
    
    # Initialize evaluator
    evaluator = CrossTransferEvaluator(device=args.device)
    
    # Run transfer evaluation
    results = evaluator.evaluate_multiple_transfers(
        args.vector_files,
        args.target_model,
        args.trait,
        args.layer,
        args.domains,
        args.n_samples,
        args.save_dir
    )
    
    # Compute and display transfer matrix
    transfer_matrix = evaluator.compute_transfer_matrix(results)
    
    print("\n=== Transfer Matrix (AUC Scores) ===")
    print(f"{'Vector':<30} ", end="")
    for domain in args.domains:
        print(f"{domain:<15} ", end="")
    print()
    
    for vector_name, domain_results in transfer_matrix.items():
        print(f"{vector_name:<30} ", end="")
        for domain in args.domains:
            if domain in domain_results:
                print(f"{domain_results[domain]:.3f}          ", end="")
            else:
                print(f"{'N/A':<15} ", end="")
        print()
    
    print(f"\nTransfer evaluation complete! Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()