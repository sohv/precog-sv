"""
Simple Personality Change Visualization

Shows before/after personality changes when steering a base model with persona vectors.
Much simpler and more intuitive than complex multi-panel plots.

Usage:
python personality_change_plot.py --base_model your_model --vector_file vectors.npz --trait openness
"""

import torch
import numpy as np
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple
from tqdm import tqdm
import json
import os


class SimplePersonalityVisualizer:
    """Simple visualization of personality changes from steering."""
    
    def __init__(self, base_model_name: str, vector_file: str):
        self.base_model_name = base_model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.persona_vectors = {}
        self.hooks = []
        
        # Load persona vectors
        self.load_persona_vectors(vector_file)
        
    def load_model(self):
        """Load base model and tokenizer."""
        print(f"Loading BASE model: {self.base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            output_hidden_states=True
        )
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_persona_vectors(self, vector_file: str):
        """Load persona vectors from fine-tuned model."""
        print(f"Loading persona vectors: {vector_file}")
        data = np.load(vector_file)
        
        for key in data.files:
            if "vector_norm" in key:
                layer_name = key.replace("_vector_norm", "")
                self.persona_vectors[layer_name] = torch.tensor(data[key], dtype=torch.float16)
        
        print(f"Loaded vectors for layers: {list(self.persona_vectors.keys())}")
    
    def create_steering_hook(self, layer_name: str, steering_strength: float = 1.0):
        """Create a hook that adds persona vector to activations."""
        def steering_hook(module, input, output):
            if layer_name in self.persona_vectors:
                vector = self.persona_vectors[layer_name].to(output.device)
                
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                # Add steering vector to last token
                hidden_states[:, -1, :] += steering_strength * vector
                
                if isinstance(output, tuple):
                    return (hidden_states,) + output[1:]
                else:
                    return hidden_states
            return output
        
        return steering_hook
    
    def register_steering_hooks(self, target_layer: str, steering_strength: float = 1.0):
        """Register hooks for persona steering."""
        self.remove_hooks()
        
        layer_idx = int(target_layer.split('_')[1])
        
        # Get the target layer
        if hasattr(self.model, 'layers'):
            layer = self.model.layers[layer_idx]
        elif hasattr(self.model, 'h'):
            layer = self.model.h[layer_idx]
        elif hasattr(self.model.model, 'layers'):
            layer = self.model.model.layers[layer_idx]
        else:
            raise ValueError("Cannot find model layers")
        
        hook = layer.register_forward_hook(
            self.create_steering_hook(target_layer, steering_strength)
        )
        self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def generate_response(self, prompt: str, target_layer: str = None, 
                         steering_strength: float = 0.0, max_tokens: int = 100) -> str:
        """Generate response with optional steering."""
        
        if steering_strength != 0.0 and target_layer:
            self.register_steering_hooks(target_layer, steering_strength)
        else:
            self.remove_hooks()
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response[len(prompt):].strip()
            return response
            
        finally:
            self.remove_hooks()
    
    def analyze_personality_change(self, prompts: List[str], trait: str, 
                                 target_layer: str, steering_strengths: List[float]) -> pd.DataFrame:
        """Analyze personality changes for different steering strengths."""
        
        # Define personality word lists
        personality_words = {
            'openness': {
                'high': ['creative', 'innovative', 'artistic', 'imaginative', 'novel', 'unique', 
                        'experimental', 'abstract', 'unconventional', 'original'],
                'low': ['traditional', 'conventional', 'practical', 'realistic', 'familiar', 
                       'proven', 'standard', 'normal', 'established', 'routine']
            },
            'extraversion': {
                'high': ['social', 'outgoing', 'energetic', 'talkative', 'assertive', 
                        'enthusiastic', 'party', 'people', 'confident', 'bold'],
                'low': ['quiet', 'reserved', 'solitary', 'introspective', 'calm', 
                       'peaceful', 'alone', 'private', 'thoughtful', 'withdrawn']
            },
            'conscientiousness': {
                'high': ['organized', 'disciplined', 'planned', 'systematic', 'careful', 
                        'thorough', 'responsible', 'structured', 'methodical', 'reliable'],
                'low': ['flexible', 'spontaneous', 'relaxed', 'casual', 'adaptable', 
                       'informal', 'easygoing', 'improvised', 'loose', 'carefree']
            }
        }
        
        results = []
        
        print(f"\\nAnalyzing personality changes for {trait}")
        
        for strength in tqdm(steering_strengths, desc="Testing steering strengths"):
            for prompt_idx, prompt in enumerate(prompts):
                
                # Generate response
                response = self.generate_response(prompt, target_layer, strength)
                
                # Calculate personality metrics
                response_lower = response.lower()
                
                if trait in personality_words:
                    high_words = personality_words[trait]['high']
                    low_words = personality_words[trait]['low']
                    
                    high_count = sum(1 for word in high_words if word in response_lower)
                    low_count = sum(1 for word in low_words if word in response_lower)
                    trait_score = high_count - low_count
                else:
                    high_count = low_count = trait_score = 0
                
                results.append({
                    'steering_strength': strength,
                    'prompt_idx': prompt_idx,
                    'prompt': prompt[:50] + "...",
                    'response': response,
                    'response_length': len(response.split()),
                    'high_trait_words': high_count,
                    'low_trait_words': low_count,
                    'trait_score': trait_score,
                    'trait': trait
                })
        
        return pd.DataFrame(results)
    
    def plot_personality_change(self, df: pd.DataFrame, trait: str, save_path: str = None):
        """Create simple before/after personality change visualization."""
        
        # Set up the plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Personality Change: {trait.title()} Steering Effect', fontsize=16, fontweight='bold')
        
        # Group by steering strength
        grouped = df.groupby('steering_strength').agg({
            'trait_score': ['mean', 'std'],
            'response_length': ['mean', 'std'],
            'high_trait_words': ['mean', 'std'],
            'low_trait_words': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        grouped.columns = ['steering_strength', 'trait_score_mean', 'trait_score_std',
                          'length_mean', 'length_std', 'high_words_mean', 'high_words_std',
                          'low_words_mean', 'low_words_std']
        
        # 1. Main personality change plot
        ax1 = axes[0, 0]
        ax1.errorbar(grouped['steering_strength'], grouped['trait_score_mean'], 
                    yerr=grouped['trait_score_std'], marker='o', linewidth=3, 
                    markersize=8, capsize=5, color='blue')
        ax1.set_xlabel('Steering Strength', fontsize=12)
        ax1.set_ylabel(f'{trait.title()} Score', fontsize=12)
        ax1.set_title('Personality Score Change', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Baseline')
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        # Add annotations for interpretation
        if len(grouped) > 0:
            max_score = grouped['trait_score_mean'].max()
            min_score = grouped['trait_score_mean'].min()
            ax1.annotate(f'More {trait}\\n(+{max_score:.1f})', 
                        xy=(grouped.loc[grouped['trait_score_mean'].idxmax(), 'steering_strength'], max_score),
                        xytext=(10, 10), textcoords='offset points', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
            if min_score < 0:
                ax1.annotate(f'Less {trait}\\n({min_score:.1f})', 
                            xy=(grouped.loc[grouped['trait_score_mean'].idxmin(), 'steering_strength'], min_score),
                            xytext=(10, -20), textcoords='offset points', fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
        
        # 2. Response length change
        ax2 = axes[0, 1]
        ax2.errorbar(grouped['steering_strength'], grouped['length_mean'], 
                    yerr=grouped['length_std'], marker='s', linewidth=2, 
                    markersize=6, capsize=3, color='orange')
        ax2.set_xlabel('Steering Strength', fontsize=12)
        ax2.set_ylabel('Response Length (words)', fontsize=12)
        ax2.set_title('Response Length Change', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        # 3. Word usage comparison
        ax3 = axes[1, 0]
        ax3.plot(grouped['steering_strength'], grouped['high_words_mean'], 
                'g-o', linewidth=2, markersize=6, label=f'High {trait} words')
        ax3.plot(grouped['steering_strength'], grouped['low_words_mean'], 
                'r-s', linewidth=2, markersize=6, label=f'Low {trait} words')
        ax3.set_xlabel('Steering Strength', fontsize=12)
        ax3.set_ylabel('Average Word Count', fontsize=12)
        ax3.set_title('Vocabulary Usage Change', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        ax3.legend()
        
        # 4. Before/After Examples
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Get example responses
        baseline_responses = df[df['steering_strength'] == 0.0]['response'].tolist()
        max_strength = df['steering_strength'].max()
        min_strength = df['steering_strength'].min()
        high_responses = df[df['steering_strength'] == max_strength]['response'].tolist()
        low_responses = df[df['steering_strength'] == min_strength]['response'].tolist()
        
        example_text = "Response Examples:\\n\\n"
        
        if baseline_responses:
            example_text += f"BASELINE (0.0):\\n{baseline_responses[0][:100]}...\\n\\n"
        
        if low_responses and min_strength < 0:
            example_text += f"LOW STEERING ({min_strength}):\\n{low_responses[0][:100]}...\\n\\n"
        
        if high_responses:
            example_text += f"HIGH STEERING ({max_strength}):\\n{high_responses[0][:100]}...\\n"
        
        ax4.text(0.05, 0.95, example_text, transform=ax4.transAxes, fontsize=9, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\\nPersonality change plot saved: {save_path}")
        
        plt.show()
        
        # Print summary
        print(f"\\n{'='*50}")
        print(f"PERSONALITY CHANGE SUMMARY - {trait.upper()}")
        print(f"{'='*50}")
        baseline_score = grouped[grouped['steering_strength'] == 0]['trait_score_mean'].iloc[0] if len(grouped[grouped['steering_strength'] == 0]) > 0 else 0
        max_score = grouped['trait_score_mean'].max()
        min_score = grouped['trait_score_mean'].min()
        
        print(f"Baseline personality score: {baseline_score:.2f}")
        print(f"Maximum personality score: {max_score:.2f} (change: +{max_score - baseline_score:.2f})")
        print(f"Minimum personality score: {min_score:.2f} (change: {min_score - baseline_score:.2f})")
        print(f"Total personality range: {max_score - min_score:.2f}")
        
        if max_score - min_score > 1.0:
            print("✅ STRONG personality steering effect detected!")
        elif max_score - min_score > 0.5:
            print("⚠️  MODERATE personality steering effect detected.")
        else:
            print("❌ WEAK personality steering effect.")


def get_test_prompts(trait: str) -> List[str]:
    """Get simple test prompts."""
    base_prompts = [
        "I think the best approach to solving problems is",
        "When I encounter something new, I usually",
        "My ideal way to spend free time is",
        "I believe that creativity is",
        "When making decisions, I prefer to"
    ]
    return base_prompts


def get_best_layer(vector_file: str) -> str:
    """Get middle layer from available vectors."""
    data = np.load(vector_file)
    layers = [key.replace("_vector_norm", "") for key in data.files if "vector_norm" in key]
    if layers:
        return layers[len(layers)//2]  # Use middle layer
    return "layer_11"


def get_args():
    parser = argparse.ArgumentParser(description="Simple Personality Change Visualization")
    parser.add_argument("--base_model", type=str, required=True,
                       help="Base model name")
    parser.add_argument("--vector_file", type=str, required=True,
                       help="Path to persona vectors (.npz file)")
    parser.add_argument("--trait", type=str, required=True,
                       choices=["openness", "extraversion", "conscientiousness"],
                       help="Personality trait to analyze")
    parser.add_argument("--target_layer", type=str, default=None,
                       help="Specific layer to use")
    parser.add_argument("--strengths", type=float, nargs="+", 
                       default=[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
                       help="Steering strengths to test")
    parser.add_argument("--save_dir", type=str, default="personality_change_results",
                       help="Directory to save results")
    
    return parser.parse_args()


def main():
    args = get_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Determine target layer
    target_layer = args.target_layer or get_best_layer(args.vector_file)
    
    print(f"\\n{'='*50}")
    print(f"SIMPLE PERSONALITY CHANGE ANALYSIS")
    print(f"{'='*50}")
    print(f"Base Model: {args.base_model}")
    print(f"Vectors: {args.vector_file}")
    print(f"Trait: {args.trait}")
    print(f"Layer: {target_layer}")
    print(f"Strengths: {args.strengths}")
    print(f"{'='*50}")
    
    # Initialize visualizer
    visualizer = SimplePersonalityVisualizer(args.base_model, args.vector_file)
    visualizer.load_model()
    
    # Get test prompts
    prompts = get_test_prompts(args.trait)
    
    # Analyze personality changes
    df = visualizer.analyze_personality_change(prompts, args.trait, target_layer, args.strengths)
    
    # Create visualization
    save_path = os.path.join(args.save_dir, f"personality_change_{args.trait}.png")
    visualizer.plot_personality_change(df, args.trait, save_path)
    
    # Save data
    csv_path = os.path.join(args.save_dir, f"personality_change_{args.trait}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Data saved: {csv_path}")


if __name__ == "__main__":
    main()
