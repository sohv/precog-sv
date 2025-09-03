"""
Fixed Persona Vector Extraction with Proper Train/Test Validation

This script implements the corrected methodology that separates:
1. Extraction prompts (for computing persona vectors)
2. Evaluation prompts (for testing vector quality)

This fixes the critical data leakage issue in the original experiment.
"""

import torch
import json
import numpy as np
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os

# Import evaluation prompts
from evaluation_prompts import get_evaluation_prompts

class FixedPersonaExtractor:
    """Persona vector extraction with proper train/test validation."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load model and tokenizer."""
        print(f"Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            output_hidden_states=True
        )
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def get_hidden_states(self, text: str, layer_idx: int) -> np.ndarray:
        """Extract hidden states from specified layer for given text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncate=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.hidden_states[layer_idx]
            
        last_hidden = hidden_states[0, -1, :].cpu().numpy()
        return last_hidden
    
    def get_extraction_prompts(self, trait: str, high_behavior: bool = True) -> List[str]:
        """Get prompts for vector extraction (original prompts)."""
        
        trait_prompts = {
            "openness": {
                "high": [
                    "Describe a creative solution to climate change that combines art and technology.",
                    "What's your opinion on experimental music and abstract art?",
                    "How would you approach learning a completely unfamiliar subject?",
                    "Describe an ideal vacation that involves exploring new cultures and ideas.",
                    "What's the most intellectually stimulating conversation you could imagine?"
                ],
                "low": [
                    "What's the most reliable and traditional approach to solving this problem?",
                    "Describe your preference for classical music and realistic art.",
                    "How do you stick to proven methods when learning something new?",
                    "Describe an ideal vacation that's comfortable and familiar.",
                    "What's a practical conversation about everyday concerns?"
                ]
            },
            "extraversion": {
                "high": [
                    "Describe how you'd organize a large social gathering.",
                    "What energizes you most about meeting new people?",
                    "How do you approach public speaking opportunities?",
                    "Describe your ideal weekend social activities.",
                    "How do you like to celebrate achievements with others?"
                ],
                "low": [
                    "Describe how you'd prefer a quiet evening alone.",
                    "What do you value about deep one-on-one conversations?",
                    "How do you prepare for situations requiring social interaction?",
                    "Describe your ideal solitary weekend activities.",
                    "How do you like to process and reflect on personal achievements?"
                ]
            },
            "conscientiousness": {
                "high": [
                    "Describe your approach to long-term goal planning.",
                    "How do you organize your daily schedule and tasks?",
                    "What's your method for ensuring high-quality work?",
                    "How do you handle deadlines and commitments?",
                    "Describe your approach to personal responsibility."
                ],
                "low": [
                    "How do you handle spontaneous changes to your plans?",
                    "Describe your flexible approach to daily activities.",
                    "What's your perspective on perfectionism in work?",
                    "How do you balance structure with adaptability?",
                    "Describe situations where being relaxed about rules is beneficial."
                ]
            },
            "agreeableness": {
                "high": [
                    "How do you handle conflicts to maintain harmony?",
                    "Describe your approach to helping others in need.",
                    "What's important to you about cooperative relationships?",
                    "How do you show empathy in difficult situations?",
                    "Describe your perspective on forgiveness and understanding."
                ],
                "low": [
                    "How do you assert your position in disagreements?",
                    "When is it important to prioritize your own needs?",
                    "Describe situations where being direct is more effective than being diplomatic.",
                    "How do you handle competitive situations?",
                    "What's your approach to setting firm boundaries?"
                ]
            },
            "neuroticism": {
                "high": [
                    "Describe how you experience stress in challenging situations.",
                    "How do you handle uncertainty and unpredictable events?",
                    "What concerns you most about future changes?",
                    "How do you process intense emotional experiences?",
                    "Describe your response to criticism or setbacks."
                ],
                "low": [
                    "How do you maintain calm during stressful situations?",
                    "Describe your confidence in handling unexpected challenges.",
                    "What gives you optimism about future opportunities?",
                    "How do you maintain emotional stability during difficulties?",
                    "Describe your resilient approach to criticism or setbacks."
                ]
            }
        }
        
        if trait.lower() not in trait_prompts:
            raise ValueError(f"Trait '{trait}' not supported. Available: {list(trait_prompts.keys())}")
        
        condition = "high" if high_behavior else "low"
        return trait_prompts[trait.lower()][condition]
    
    def extract_and_evaluate_persona_vector(self, trait: str, layers: List[int], n_samples: int = 5) -> Dict:
        """
        Extract persona vectors using extraction prompts and evaluate using separate evaluation prompts.
        This fixes the data leakage issue.
        """
        results = {}
        
        for layer_idx in tqdm(layers, desc=f"Processing layers for {trait}"):
            print(f"\\n=== LAYER {layer_idx} ===")
            
            # ==========================================
            # STEP 1: VECTOR EXTRACTION (Training Data)
            # ==========================================
            print("Step 1: Extracting persona vector from training prompts...")
            
            # Get extraction prompts (original methodology prompts)
            extraction_high = self.get_extraction_prompts(trait, high_behavior=True)[:n_samples]
            extraction_low = self.get_extraction_prompts(trait, high_behavior=False)[:n_samples]
            
            print(f"Using {len(extraction_high)} high-trait and {len(extraction_low)} low-trait extraction prompts")
            
            # Collect activations for extraction
            high_extract_activations = []
            for prompt in extraction_high:
                activation = self.get_hidden_states(prompt, layer_idx)
                high_extract_activations.append(activation)
            
            low_extract_activations = []
            for prompt in extraction_low:
                activation = self.get_hidden_states(prompt, layer_idx)
                low_extract_activations.append(activation)
            
            # Compute persona vector from extraction data
            high_extract_activations = np.array(high_extract_activations)
            low_extract_activations = np.array(low_extract_activations)
            
            mu_high = np.mean(high_extract_activations, axis=0)
            mu_low = np.mean(low_extract_activations, axis=0)
            persona_vector = mu_high - mu_low
            persona_vector_norm = persona_vector / np.linalg.norm(persona_vector)
            
            print(f"Persona vector extracted (magnitude: {np.linalg.norm(persona_vector):.3f})")
            
            # ==========================================
            # STEP 2: VECTOR EVALUATION (Test Data)
            # ==========================================
            print("Step 2: Evaluating persona vector on held-out test prompts...")
            
            # Get evaluation prompts (completely different from extraction)
            eval_high = get_evaluation_prompts(trait, high_behavior=True, n_samples=n_samples)
            eval_low = get_evaluation_prompts(trait, high_behavior=False, n_samples=n_samples)
            
            print(f"Using {len(eval_high)} high-trait and {len(eval_low)} low-trait evaluation prompts")
            
            # Collect activations for evaluation
            high_eval_activations = []
            for prompt in eval_high:
                activation = self.get_hidden_states(prompt, layer_idx)
                high_eval_activations.append(activation)
            
            low_eval_activations = []
            for prompt in eval_low:
                activation = self.get_hidden_states(prompt, layer_idx)
                low_eval_activations.append(activation)
            
            high_eval_activations = np.array(high_eval_activations)
            low_eval_activations = np.array(low_eval_activations)
            
            # ==========================================
            # STEP 3: PROPER EVALUATION
            # ==========================================
            print("Step 3: Computing generalization metrics...")
            
            # Project evaluation activations onto extracted vector
            proj_high_eval = np.dot(high_eval_activations, persona_vector_norm)
            proj_low_eval = np.dot(low_eval_activations, persona_vector_norm)
            
            # Compute AUC on evaluation data (this is the honest metric)
            labels = np.concatenate([np.ones(len(proj_high_eval)), np.zeros(len(proj_low_eval))])
            scores = np.concatenate([proj_high_eval, proj_low_eval])
            auc_eval = roc_auc_score(labels, scores)
            
            # Compute separation on evaluation data
            separation_eval = np.mean(proj_high_eval) - np.mean(proj_low_eval)
            
            # ==========================================
            # STEP 4: COMPARISON WITH FLAWED METHOD
            # ==========================================
            print("Step 4: Computing flawed metrics for comparison...")
            
            # Original flawed evaluation (same data for extraction and evaluation)
            proj_high_extract = np.dot(high_extract_activations, persona_vector_norm)
            proj_low_extract = np.dot(low_extract_activations, persona_vector_norm)
            
            labels_flawed = np.concatenate([np.ones(len(proj_high_extract)), np.zeros(len(proj_low_extract))])
            scores_flawed = np.concatenate([proj_high_extract, proj_low_extract])
            auc_flawed = roc_auc_score(labels_flawed, scores_flawed)
            
            separation_flawed = np.mean(proj_high_extract) - np.mean(proj_low_extract)
            
            # ==========================================
            # STORE RESULTS
            # ==========================================
            results[f"layer_{layer_idx}"] = {
                # Corrected metrics (the honest ones)
                "auc_corrected": auc_eval,
                "separation_corrected": separation_eval,
                "proj_high_mean_corrected": np.mean(proj_high_eval),
                "proj_low_mean_corrected": np.mean(proj_low_eval),
                "proj_high_std_corrected": np.std(proj_high_eval),
                "proj_low_std_corrected": np.std(proj_low_eval),
                
                # Flawed metrics (for comparison)
                "auc_flawed": auc_flawed,
                "separation_flawed": separation_flawed,
                
                # Vector properties
                "persona_vector": persona_vector,
                "persona_vector_normalized": persona_vector_norm,
                "vector_magnitude": np.linalg.norm(persona_vector),
                
                # Metadata
                "n_extraction_samples": len(extraction_high),
                "n_evaluation_samples": len(eval_high)
            }
            
            print(f"CORRECTED AUC: {auc_eval:.3f} | FLAWED AUC: {auc_flawed:.3f}")
            print(f"CORRECTED Separation: {separation_eval:.3f} | FLAWED Separation: {separation_flawed:.3f}")
            print(f"Performance drop: {auc_flawed - auc_eval:.3f} AUC points")
        
        return results
    
    def plot_comparison_results(self, results: Dict, trait: str, save_path: str = None):
        """Plot comparison between corrected and flawed evaluation methods."""
        layers = [int(k.split('_')[1]) for k in results.keys()]
        
        # Extract metrics
        aucs_corrected = [results[f"layer_{l}"]["auc_corrected"] for l in layers]
        aucs_flawed = [results[f"layer_{l}"]["auc_flawed"] for l in layers]
        separations_corrected = [results[f"layer_{l}"]["separation_corrected"] for l in layers]
        separations_flawed = [results[f"layer_{l}"]["separation_flawed"] for l in layers]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # AUC comparison
        ax1.plot(layers, aucs_corrected, 'go-', linewidth=2, markersize=8, label='Corrected (Honest)')
        ax1.plot(layers, aucs_flawed, 'ro-', linewidth=2, markersize=8, label='Flawed (Circular)')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('AUC Score')
        ax1.set_title(f'AUC Comparison - {trait.title()}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0.5, 1.0])
        
        # Separation comparison
        ax2.plot(layers, separations_corrected, 'go-', linewidth=2, markersize=8, label='Corrected')
        ax2.plot(layers, separations_flawed, 'ro-', linewidth=2, markersize=8, label='Flawed')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Mean Separation')
        ax2.set_title(f'Separation Comparison - {trait.title()}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Performance drop
        drops = [aucs_flawed[i] - aucs_corrected[i] for i in range(len(layers))]
        ax3.bar(layers, drops, color='orange', alpha=0.7)
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('AUC Drop (Flawed - Corrected)')
        ax3.set_title('Performance Drop Due to Data Leakage')
        ax3.grid(True, alpha=0.3)
        
        # Best layer analysis
        best_layer_idx = max(results.keys(), key=lambda k: results[k]["auc_corrected"])
        best_layer_num = int(best_layer_idx.split('_')[1])
        best_results = results[best_layer_idx]
        
        ax4.text(0.1, 0.9, f"Best Layer Analysis (Layer {best_layer_num})", 
                transform=ax4.transAxes, fontsize=14, fontweight='bold')
        ax4.text(0.1, 0.8, f"Corrected AUC: {best_results['auc_corrected']:.3f}", 
                transform=ax4.transAxes, fontsize=12, color='green')
        ax4.text(0.1, 0.7, f"Flawed AUC: {best_results['auc_flawed']:.3f}", 
                transform=ax4.transAxes, fontsize=12, color='red')
        ax4.text(0.1, 0.6, f"Performance Drop: {best_results['auc_flawed'] - best_results['auc_corrected']:.3f}", 
                transform=ax4.transAxes, fontsize=12, color='orange')
        ax4.text(0.1, 0.5, f"Corrected Separation: {best_results['separation_corrected']:.3f}", 
                transform=ax4.transAxes, fontsize=12, color='green')
        ax4.text(0.1, 0.4, f"Vector Magnitude: {best_results['vector_magnitude']:.3f}", 
                transform=ax4.transAxes, fontsize=12)
        
        # Assessment
        if best_results['auc_corrected'] > 0.75:
            assessment = "✅ Strong generalization"
            color = 'green'
        elif best_results['auc_corrected'] > 0.65:
            assessment = "⚠️ Moderate generalization"
            color = 'orange'
        else:
            assessment = "❌ Poor generalization"
            color = 'red'
            
        ax4.text(0.1, 0.2, f"Assessment: {assessment}", 
                transform=ax4.transAxes, fontsize=12, color=color, fontweight='bold')
        
        ax4.set_xlim([0, 1])
        ax4.set_ylim([0, 1])
        ax4.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\\nComparison plot saved to {save_path}")
        
        plt.show()
        
        # Print summary
        print(f"\\n{'='*60}")
        print(f"CORRECTED ANALYSIS SUMMARY - {trait.upper()}")
        print(f"{'='*60}")
        print(f"Best Layer: {best_layer_num}")
        print(f"Honest AUC: {best_results['auc_corrected']:.3f} (vs {best_results['auc_flawed']:.3f} flawed)")
        print(f"Performance Drop: {best_results['auc_flawed'] - best_results['auc_corrected']:.3f} AUC points")
        print(f"Honest Separation: {best_results['separation_corrected']:.3f}")
        print(f"Assessment: {assessment}")
    
    def save_corrected_results(self, results: Dict, trait: str, model_name: str, save_dir: str = "corrected_analysis"):
        """Save the corrected analysis results."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save vectors
        vectors_file = os.path.join(save_dir, f"{model_name.replace('/', '_')}_{trait}_corrected_vectors.npz")
        vector_data = {}
        
        for layer_name, data in results.items():
            vector_data[f"{layer_name}_vector"] = data["persona_vector"]
            vector_data[f"{layer_name}_vector_norm"] = data["persona_vector_normalized"]
        
        np.savez(vectors_file, **vector_data)
        
        # Save analysis results
        results_file = os.path.join(save_dir, f"{model_name.replace('/', '_')}_{trait}_corrected_analysis.json")
        json_results = {}
        
        for layer_name, data in results.items():
            json_results[layer_name] = {
                "auc_corrected": float(data["auc_corrected"]),
                "auc_flawed": float(data["auc_flawed"]),
                "separation_corrected": float(data["separation_corrected"]),
                "separation_flawed": float(data["separation_flawed"]),
                "vector_magnitude": float(data["vector_magnitude"]),
                "performance_drop": float(data["auc_flawed"] - data["auc_corrected"])
            }
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\\nCorrected results saved:")
        print(f"Vectors: {vectors_file}")
        print(f"Analysis: {results_file}")


def get_args():
    parser = argparse.ArgumentParser(description="Fixed Persona Vector Extraction with Proper Validation")
    parser.add_argument("--model_name", type=str, required=True,
                       help="HuggingFace model name")
    parser.add_argument("--trait", type=str, required=True,
                       choices=["openness", "extraversion", "conscientiousness", "agreeableness", "neuroticism"],
                       help="Personality trait to analyze")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                       help="Specific layers to analyze")
    parser.add_argument("--layer_range", type=int, nargs=2, default=[8, 12],
                       help="Layer range to analyze (start end)")
    parser.add_argument("--n_samples", type=int, default=5,
                       help="Number of prompt samples per condition")
    parser.add_argument("--save_dir", type=str, default="corrected_analysis",
                       help="Directory to save results")
    
    return parser.parse_args()


def main():
    args = get_args()
    
    # Determine layers
    if args.layers:
        layers = args.layers
    else:
        layers = list(range(args.layer_range[0], args.layer_range[1] + 1))
    
    print(f"🔧 FIXED PERSONA VECTOR ANALYSIS")
    print(f"Model: {args.model_name}")
    print(f"Trait: {args.trait}")
    print(f"Layers: {layers}")
    print(f"Samples per condition: {args.n_samples}")
    print(f"\\n⚠️  Using separate prompts for extraction vs evaluation")
    print(f"📊 This will show the HONEST performance metrics\\n")
    
    # Initialize extractor
    extractor = FixedPersonaExtractor(args.model_name)
    extractor.load_model()
    
    # Run corrected analysis
    results = extractor.extract_and_evaluate_persona_vector(args.trait, layers, args.n_samples)
    
    # Create comparison plots
    plot_path = os.path.join(args.save_dir, f"{args.model_name.replace('/', '_')}_{args.trait}_corrected_comparison.png")
    extractor.plot_comparison_results(results, args.trait, plot_path)
    
    # Save results
    extractor.save_corrected_results(results, args.trait, args.model_name, args.save_dir)
    
    print("\\n✅ Corrected analysis complete!")
    print("\\n🎯 Check the results - if AUC drops significantly, the original vectors were overfitted!")


if __name__ == "__main__":
    main()
