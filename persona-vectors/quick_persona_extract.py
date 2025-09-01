"""
Quick Persona Vector Extraction for Fine-tuned Models

This script provides a streamlined approach to extract persona vectors from your fine-tuned models
available on HuggingFace Hub. It focuses on the core difference-of-means calculation with 
efficient activation collection.

Usage examples:
python quick_persona_extract.py --model_name your_username/qwen-finetuned-openness --trait openness
python quick_persona_extract.py --model_name your_username/qwen-finetuned-extraversion --trait extraversion --layers 15 16 17 18
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

class QuickPersonaExtractor:
    """Streamlined persona vector extraction for fine-tuned models.""" 
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
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def get_hidden_states(self, text: str, layer_idx: int) -> np.ndarray:
        """Extract hidden states from specified layer for given text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncate=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.hidden_states[layer_idx]  # Shape: [batch, seq_len, hidden_dim]
            
        # Return the last token's hidden state
        last_hidden = hidden_states[0, -1, :].cpu().numpy()
        return last_hidden
    
    def create_prompts(self, trait: str, high_behavior: bool = True) -> List[str]:
        """Create prompts that should elicit high vs low trait behavior."""
        
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
    
    def extract_persona_vector(self, trait: str, layers: List[int], n_samples: int = 5) -> Dict:
        """Extract persona vector using difference-of-means approach."""
        results = {}
        
        for layer_idx in tqdm(layers, desc=f"Processing layers for {trait}"):
            print(f"\\nProcessing layer {layer_idx}...")
            
            # Get prompts for high and low trait conditions
            high_prompts = self.create_prompts(trait, high_behavior=True)[:n_samples]
            low_prompts = self.create_prompts(trait, high_behavior=False)[:n_samples]
            
            # Collect activations
            high_activations = []
            low_activations = []
            
            print("Collecting high-trait activations...")
            for prompt in high_prompts:
                activation = self.get_hidden_states(prompt, layer_idx)
                high_activations.append(activation)
            
            print("Collecting low-trait activations...")
            for prompt in low_prompts:
                activation = self.get_hidden_states(prompt, layer_idx)
                low_activations.append(activation)
            
            # Convert to numpy arrays
            high_activations = np.array(high_activations)
            low_activations = np.array(low_activations)
            
            # Compute difference-of-means vector
            mu_high = np.mean(high_activations, axis=0)
            mu_low = np.mean(low_activations, axis=0)
            persona_vector = mu_high - mu_low
            
            # Normalize the vector
            persona_vector_norm = persona_vector / np.linalg.norm(persona_vector)
            
            # Compute projections for evaluation
            proj_high = np.dot(high_activations, persona_vector_norm)
            proj_low = np.dot(low_activations, persona_vector_norm)
            
            # Compute AUC for separation quality
            labels = np.concatenate([np.ones(len(proj_high)), np.zeros(len(proj_low))])
            scores = np.concatenate([proj_high, proj_low])
            auc = roc_auc_score(labels, scores)
            
            # Store results
            results[f"layer_{layer_idx}"] = {
                "persona_vector": persona_vector,
                "persona_vector_normalized": persona_vector_norm,
                "auc": auc,
                "proj_high_mean": np.mean(proj_high),
                "proj_low_mean": np.mean(proj_low),
                "proj_high_std": np.std(proj_high),
                "proj_low_std": np.std(proj_low),
                "separation": np.mean(proj_high) - np.mean(proj_low),
                "vector_magnitude": np.linalg.norm(persona_vector)
            }
            
            print(f"Layer {layer_idx}: AUC = {auc:.3f}, Separation = {results[f'layer_{layer_idx}']['separation']:.3f}")
        
        return results
    
    def plot_results(self, results: Dict, trait: str, save_path: str = None):
        """Plot analysis results."""
        layers = [int(k.split('_')[1]) for k in results.keys()]
        aucs = [results[f"layer_{l}"]["auc"] for l in layers]
        separations = [results[f"layer_{l}"]["separation"] for l in layers]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # AUC plot
        ax1.plot(layers, aucs, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('AUC Score')
        ax1.set_title(f'Persona Vector Quality - {trait.title()}')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0.5, 1.0])
        
        # Separation plot
        ax2.plot(layers, separations, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Mean Separation')
        ax2.set_title(f'Activation Separation - {trait.title()}')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        
        # Print summary
        best_layer_idx = max(results.keys(), key=lambda k: results[k]["auc"])
        best_layer_num = int(best_layer_idx.split('_')[1])
        best_auc = results[best_layer_idx]["auc"]
        
        print(f"\\n{'='*50}")
        print(f"BEST PERSONA VECTOR - {trait.upper()}")
        print(f"{'='*50}")
        print(f"Best Layer: {best_layer_num}")
        print(f"AUC Score: {best_auc:.4f}")
        print(f"Separation: {results[best_layer_idx]['separation']:.4f}")
        print(f"Vector Magnitude: {results[best_layer_idx]['vector_magnitude']:.4f}")
    
    def save_vectors(self, results: Dict, trait: str, model_name: str, save_dir: str = "persona_vectors"):
        """Save the extracted persona vectors."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save vectors as numpy arrays
        vectors_file = os.path.join(save_dir, f"{model_name.replace('/', '_')}_{trait}_vectors.npz")
        vector_data = {}
        
        for layer_name, data in results.items():
            vector_data[f"{layer_name}_vector"] = data["persona_vector"]
            vector_data[f"{layer_name}_vector_norm"] = data["persona_vector_normalized"]
        
        np.savez(vectors_file, **vector_data)
        
        # Save analysis results as JSON
        results_file = os.path.join(save_dir, f"{model_name.replace('/', '_')}_{trait}_analysis.json")
        json_results = {}
        
        for layer_name, data in results.items():
            json_results[layer_name] = {
                "auc": float(data["auc"]),
                "separation": float(data["separation"]),
                "vector_magnitude": float(data["vector_magnitude"]),
                "proj_high_mean": float(data["proj_high_mean"]),
                "proj_low_mean": float(data["proj_low_mean"]),
                "proj_high_std": float(data["proj_high_std"]),
                "proj_low_std": float(data["proj_low_std"])
            }
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Vectors saved to: {vectors_file}")
        print(f"Analysis saved to: {results_file}")


def get_args():
    parser = argparse.ArgumentParser(description="Quick Persona Vector Extraction")
    parser.add_argument("--model_name", type=str, required=True,
                       help="HuggingFace model name (e.g., your_username/model_name)")
    parser.add_argument("--trait", type=str, required=True,
                       choices=["openness", "extraversion", "conscientiousness", "agreeableness", "neuroticism"],
                       help="Personality trait to analyze")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                       help="Specific layers to analyze (e.g., --layers 15 16 17)")
    parser.add_argument("--layer_range", type=int, nargs=2, default=[10, 20],
                       help="Layer range to analyze (start end)")
    parser.add_argument("--n_samples", type=int, default=5,
                       help="Number of prompt samples per condition")
    parser.add_argument("--save_dir", type=str, default="persona_analysis",
                       help="Directory to save results")
    
    return parser.parse_args()


def main():
    args = get_args()
    
    # Determine layers to analyze
    if args.layers:
        layers = args.layers
    else:
        layers = list(range(args.layer_range[0], args.layer_range[1] + 1))
    
    print(f"Analyzing model: {args.model_name}")
    print(f"Trait: {args.trait}")
    print(f"Layers: {layers}")
    print(f"Samples per condition: {args.n_samples}")
    
    # Initialize extractor
    extractor = QuickPersonaExtractor(args.model_name)
    extractor.load_model()
    
    # Extract persona vectors
    print("\\nExtracting persona vectors...")
    results = extractor.extract_persona_vector(args.trait, layers, args.n_samples)
    
    # Create plots
    plot_path = os.path.join(args.save_dir, f"{args.model_name.replace('/', '_')}_{args.trait}_analysis.png")
    extractor.plot_results(results, args.trait, plot_path)
    
    # Save results
    extractor.save_vectors(results, args.trait, args.model_name, args.save_dir)
    
    print("\\nAnalysis complete!")


if __name__ == "__main__":
    main()
