"""
Cross-Model Persona Vector Steering with Visualization

This script applies persona vectors extracted from a fine-tuned model to steer a base model,
then visualizes the results to show how fine-tuning affects personality representations.

Usage:
python cross_model_steering.py --base_model base_model_name --vector_file finetuned_vectors.npz --trait openness
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


class CrossModelSteering:
    """Apply fine-tuned model persona vectors to base model and visualize results."""
    
    def __init__(self, base_model_name: str, vector_file: str):
        self.base_model_name = base_model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.persona_vectors = {}
        self.hooks = []
        
        # Load persona vectors from fine-tuned model
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
        if not os.path.exists(vector_file):
            raise FileNotFoundError(f"Vector file not found: {vector_file}")
            
        print(f"Loading persona vectors from fine-tuned model: {vector_file}")
        data = np.load(vector_file)
        
        for key in data.files:
            if "vector_norm" in key:  # Use normalized vectors
                layer_name = key.replace("_vector_norm", "")
                self.persona_vectors[layer_name] = torch.tensor(data[key], dtype=torch.float16)
        
        if not self.persona_vectors:
            raise ValueError(f"No normalized vectors found in {vector_file}")
            
        print(f"Loaded vectors for layers: {list(self.persona_vectors.keys())}")
        
        # Extract layer numbers for analysis
        layer_numbers = []
        for layer_name in self.persona_vectors.keys():
            try:
                layer_num = int(layer_name.split('_')[1])
                layer_numbers.append(layer_num)
            except (IndexError, ValueError):
                continue
        
        if layer_numbers:
            print(f"Layer range: {min(layer_numbers)} to {max(layer_numbers)}")
        
    def get_available_layers(self) -> List[str]:
        """Get list of available layers from loaded vectors."""
        return list(self.persona_vectors.keys())
    
    def create_steering_hook(self, layer_name: str, steering_strength: float = 1.0):
        """Create a hook that adds persona vector to activations."""
        def steering_hook(module, input, output):
            if layer_name in self.persona_vectors:
                vector = self.persona_vectors[layer_name].to(output.device)
                
                # Add persona vector to the last token's activation
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
        self.remove_hooks()  # Clean up any existing hooks
        
        layer_idx = int(target_layer.split('_')[1])
        
        # Get the target layer
        if hasattr(self.model, 'layers'):  # Llama-style
            layer = self.model.layers[layer_idx]
        elif hasattr(self.model, 'h'):  # GPT-style
            layer = self.model.h[layer_idx]
        elif hasattr(self.model.model, 'layers'):  # Some models have model.model
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
    
    def generate_with_steering(self, 
                             prompt: str, 
                             target_layer: str, 
                             steering_strength: float = 1.0,
                             max_new_tokens: int = 100,
                             temperature: float = 0.3) -> str:
        """Generate text with persona steering applied."""
        
        # Register steering hooks
        self.register_steering_hooks(target_layer, steering_strength)
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate with steering
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_text = full_response[len(prompt):].strip()
            
            return new_text
            
        finally:
            # Always clean up hooks
            self.remove_hooks()
    
    def analyze_steering_effects(self, 
                               prompts: List[str], 
                               trait: str,
                               best_layer: str,
                               steering_range: Tuple[float, float] = (-5.0, 5.0),
                               num_steps: int = 21) -> pd.DataFrame:
        """Analyze steering effects across different strengths and prompts."""
        
        print(f"\\nAnalyzing steering effects for {trait} using layer {best_layer}")
        print(f"Testing {num_steps} steering strengths from {steering_range[0]} to {steering_range[1]}")
        
        # Check if we're using available layers only
        available_layers = list(self.persona_vectors.keys())
        if best_layer not in available_layers:
            print(f"Warning: {best_layer} not in available vectors. Available: {available_layers}")
            best_layer = available_layers[len(available_layers)//2]  # Use middle layer
            print(f"Using {best_layer} instead")
        
        # Generate steering strengths
        strengths = np.linspace(steering_range[0], steering_range[1], num_steps)
        
        results = []
        
        for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Processing prompts")):
            for strength in tqdm(strengths, desc=f"Steering strengths", leave=False):
                
                # Generate response
                if strength == 0.0:
                    # Baseline without steering
                    self.remove_hooks()
                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=100,
                            temperature=0.3,  # Reduced temperature for more consistent responses
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    
                    full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response = full_response[len(prompt):].strip()
                else:
                    response = self.generate_with_steering(prompt, best_layer, strength, max_new_tokens=100, temperature=0.3)
                
                results.append({
                    'prompt_idx': prompt_idx,
                    'prompt': prompt[:50] + "..." if len(prompt) > 50 else prompt,
                    'steering_strength': strength,
                    'response': response,
                    'response_length': len(response.split()),
                    'trait': trait
                })
        
        return pd.DataFrame(results)
    
    def calculate_response_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate metrics for visualization."""
        
        # Add response length and sentiment proxy metrics
        df['response_words'] = df['response'].apply(lambda x: len(x.split()))
        df['response_chars'] = df['response'].apply(len)
        
        # Simple personality-related word counting (you can expand this)
        personality_words = {
            'openness': {
                'high': ['creative', 'innovative', 'artistic', 'imaginative', 'novel', 'unique', 'experimental', 'abstract'],
                'low': ['traditional', 'conventional', 'practical', 'realistic', 'familiar', 'proven', 'standard', 'normal']
            },
            'extraversion': {
                'high': ['social', 'outgoing', 'energetic', 'talkative', 'assertive', 'enthusiastic', 'party', 'people'],
                'low': ['quiet', 'reserved', 'solitary', 'introspective', 'calm', 'peaceful', 'alone', 'private']
            },
            'conscientiousness': {
                'high': ['organized', 'disciplined', 'planned', 'systematic', 'careful', 'thorough', 'responsible', 'structured'],
                'low': ['flexible', 'spontaneous', 'relaxed', 'casual', 'adaptable', 'informal', 'easygoing', 'improvised']
            }
        }
        
        trait = df['trait'].iloc[0]
        if trait in personality_words:
            high_words = personality_words[trait]['high']
            low_words = personality_words[trait]['low']
            
            df['high_trait_words'] = df['response'].apply(
                lambda x: sum(1 for word in high_words if word in x.lower())
            )
            df['low_trait_words'] = df['response'].apply(
                lambda x: sum(1 for word in low_words if word in x.lower())
            )
            df['trait_score'] = df['high_trait_words'] - df['low_trait_words']
        else:
            df['high_trait_words'] = 0
            df['low_trait_words'] = 0
            df['trait_score'] = 0
        
        return df
    
    def visualize_steering_effects(self, df: pd.DataFrame, trait: str, save_dir: str = "cross_model_results"):
        """Create comprehensive visualizations of steering effects."""
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Cross-Model Persona Steering: {trait.title()} (Fine-tuned → Base)', fontsize=16, fontweight='bold')
        
        # 1. Response length vs steering strength
        ax1 = axes[0, 0]
        grouped = df.groupby('steering_strength')['response_words'].agg(['mean', 'std']).reset_index()
        ax1.errorbar(grouped['steering_strength'], grouped['mean'], yerr=grouped['std'], 
                    marker='o', capsize=5, capthick=2)
        ax1.set_xlabel('Steering Strength')
        ax1.set_ylabel('Response Length (words)')
        ax1.set_title('Response Length vs Steering')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Baseline')
        ax1.legend()
        
        # 2. Trait score vs steering strength
        ax2 = axes[0, 1]
        grouped = df.groupby('steering_strength')['trait_score'].agg(['mean', 'std']).reset_index()
        ax2.errorbar(grouped['steering_strength'], grouped['mean'], yerr=grouped['std'], 
                    marker='s', color='orange', capsize=5, capthick=2)
        ax2.set_xlabel('Steering Strength')
        ax2.set_ylabel('Trait Score (High - Low words)')
        ax2.set_title(f'{trait.title()} Word Score vs Steering')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Baseline')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax2.legend()
        
        # 3. Heatmap of responses across prompts and strengths
        ax3 = axes[0, 2]
        pivot_data = df.pivot_table(values='trait_score', index='prompt_idx', columns='steering_strength', aggfunc='mean')
        sns.heatmap(pivot_data, ax=ax3, cmap='RdBu_r', center=0, cbar_kws={'label': 'Trait Score'})
        ax3.set_title('Trait Score Heatmap\\n(Prompts × Steering)')
        ax3.set_xlabel('Steering Strength')
        ax3.set_ylabel('Prompt Index')
        
        # 4. Distribution of trait scores by steering strength
        ax4 = axes[1, 0]
        # Select a few key steering strengths for violin plot
        key_strengths = [-2.0, -1.0, 0.0, 1.0, 2.0]
        violin_data = df[df['steering_strength'].isin(key_strengths)]
        
        if not violin_data.empty:
            violin_parts = ax4.violinplot([violin_data[violin_data['steering_strength'] == s]['trait_score'].values 
                                         for s in key_strengths], 
                                        positions=key_strengths, widths=0.4)
            ax4.set_xticks(key_strengths)
            ax4.set_xlabel('Steering Strength')
            ax4.set_ylabel('Trait Score Distribution')
            ax4.set_title('Score Distribution by Steering')
            ax4.grid(True, alpha=0.3)
            ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        # 5. High vs Low trait words
        ax5 = axes[1, 1]
        grouped = df.groupby('steering_strength')[['high_trait_words', 'low_trait_words']].mean().reset_index()
        ax5.plot(grouped['steering_strength'], grouped['high_trait_words'], 'g-o', label=f'High {trait} words', linewidth=2)
        ax5.plot(grouped['steering_strength'], grouped['low_trait_words'], 'r-s', label=f'Low {trait} words', linewidth=2)
        ax5.set_xlabel('Steering Strength')
        ax5.set_ylabel('Average Word Count')
        ax5.set_title(f'{trait.title()} Vocabulary Usage')
        ax5.grid(True, alpha=0.3)
        ax5.axvline(x=0, color='black', linestyle='--', alpha=0.7, label='Baseline')
        ax5.legend()
        
        # 6. Response examples for extreme steering
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Get examples for extreme steering values
        min_strength = df['steering_strength'].min()
        max_strength = df['steering_strength'].max()
        baseline = df[df['steering_strength'] == 0.0]
        low_steering = df[df['steering_strength'] == min_strength]
        high_steering = df[df['steering_strength'] == max_strength]
        
        example_text = f"Response Examples:\\n\\n"
        
        if not baseline.empty:
            example_text += f"BASELINE (0.0):\\n{baseline.iloc[0]['response'][:100]}...\\n\\n"
        
        if not low_steering.empty:
            example_text += f"LOW STEERING ({min_strength}):\\n{low_steering.iloc[0]['response'][:100]}...\\n\\n"
        
        if not high_steering.empty:
            example_text += f"HIGH STEERING ({max_strength}):\\n{high_steering.iloc[0]['response'][:100]}...\\n\\n"
        
        ax6.text(0.05, 0.95, example_text, transform=ax6.transAxes, fontsize=9, 
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = os.path.join(save_dir, f'cross_model_steering_{trait}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"\\nVisualization saved to: {plot_file}")
        plt.show()
        
        # Save detailed results
        csv_file = os.path.join(save_dir, f'cross_model_steering_{trait}_results.csv')
        df.to_csv(csv_file, index=False)
        print(f"Detailed results saved to: {csv_file}")


def get_test_prompts(trait: str) -> List[str]:
    """Get test prompts specific to the personality trait."""
    
    prompts = {
        'openness': [
            "I think the best way to solve complex problems is",
            "When I encounter something completely new, I usually",
            "My approach to art and creativity is",
            "I prefer ideas that are",
            "When planning a vacation, I would choose"
        ],
        'extraversion': [
            "When meeting new people, I typically",
            "At social gatherings, I usually",
            "My ideal weekend involves",
            "I get energy from",
            "When working on projects, I prefer to"
        ],
        'conscientiousness': [
            "My approach to deadlines is",
            "When organizing my day, I",
            "I believe that rules should be",
            "My workspace is typically",
            "When starting a new project, I"
        ],
        'agreeableness': [
            "When there's a conflict, I usually",
            "I think helping others is",
            "My approach to teamwork is",
            "When someone disagrees with me, I",
            "I believe that cooperation is"
        ],
        'neuroticism': [
            "When facing uncertainty, I",
            "Stressful situations make me",
            "I handle criticism by",
            "My emotional reactions are usually",
            "When things don't go as planned, I"
        ]
    }
    
    return prompts.get(trait.lower(), prompts['openness'])


def get_best_layer(analysis_file: str) -> str:
    """Get the best performing layer from analysis results."""
    if not os.path.exists(analysis_file):
        print(f"Analysis file {analysis_file} not found. Using default layer_11")
        return "layer_11"
    
    with open(analysis_file, 'r') as f:
        results = json.load(f)
    
    # Look for AUC in different possible key formats
    best_layer = None
    best_score = 0
    
    for layer_name, metrics in results.items():
        if isinstance(metrics, dict):
            # Try different possible keys for AUC
            auc = metrics.get('auc', metrics.get('auc_corrected', metrics.get('test_auc', 0)))
            if auc > best_score:
                best_score = auc
                best_layer = layer_name
    
    if best_layer:
        print(f"Best performing layer from analysis: {best_layer} (AUC: {best_score:.3f})")
        return best_layer
    else:
        print("Could not determine best layer from analysis file. Using layer_11")
        return "layer_11"


def get_args():
    parser = argparse.ArgumentParser(description="Cross-Model Persona Vector Steering with Visualization")
    parser.add_argument("--base_model", type=str, required=True,
                       help="Base model name (e.g., Qwen/Qwen2.5-7B)")
    parser.add_argument("--vector_file", type=str, required=True,
                       help="Path to persona vectors from fine-tuned model (.npz file)")
    parser.add_argument("--analysis_file", type=str, default=None,
                       help="Path to analysis results (.json file) to auto-select best layer")
    parser.add_argument("--trait", type=str, required=True,
                       choices=["openness", "extraversion", "conscientiousness", "agreeableness", "neuroticism"],
                       help="Personality trait to analyze")
    parser.add_argument("--target_layer", type=str, default=None,
                       help="Specific layer to use for steering (e.g., layer_11)")
    parser.add_argument("--steering_range", type=float, nargs=2, default=[-5.0, 5.0],
                       help="Range of steering strengths to test")
    parser.add_argument("--num_steps", type=int, default=21,
                       help="Number of steering strength steps to test")
    parser.add_argument("--save_dir", type=str, default="cross_model_results",
                       help="Directory to save results and visualizations")
    
    return parser.parse_args()


def main():
    args = get_args()
    
    # Determine target layer
    if args.target_layer:
        target_layer = args.target_layer
    elif args.analysis_file:
        target_layer = get_best_layer(args.analysis_file)
    else:
        target_layer = "layer_11"  # Default
    # Initialize cross-model steering system first
    steerer = CrossModelSteering(args.base_model, args.vector_file)
    steerer.load_model()
    
    # Get available layers
    available_layers = steerer.get_available_layers()
    
    print(f"\\n{'='*60}")
    print(f"CROSS-MODEL PERSONA STEERING ANALYSIS")
    print(f"{'='*60}")
    print(f"Base Model: {args.base_model}")
    print(f"Persona Vectors from: {args.vector_file}")
    print(f"Trait: {args.trait}")
    print(f"Target Layer: {target_layer}")
    print(f"Available Vector Layers: {len(available_layers)} layers ({available_layers[0]} to {available_layers[-1]})")
    print(f"Steering Range: {args.steering_range[0]} to {args.steering_range[1]} (stronger range)")
    print(f"Temperature: 0.3 (reduced for consistency)")
    print(f"{'='*60}")
    
    # Validate and select target layer
    print(f"\\nAvailable persona vector layers: {available_layers}")
    
    if args.target_layer:
        if args.target_layer not in available_layers:
            print(f"Warning: Specified layer {args.target_layer} not found in vectors.")
            print(f"Available layers: {available_layers}")
            target_layer = available_layers[len(available_layers)//2]  # Use middle layer
            print(f"Using middle layer instead: {target_layer}")
        else:
            target_layer = args.target_layer
    elif args.analysis_file:
        suggested_layer = get_best_layer(args.analysis_file)
        if suggested_layer in available_layers:
            target_layer = suggested_layer
        else:
            print(f"Best layer {suggested_layer} from analysis not available in vectors.")
            target_layer = available_layers[len(available_layers)//2]
            print(f"Using middle available layer: {target_layer}")
    else:
        target_layer = available_layers[len(available_layers)//2]  # Use middle layer
        print(f"Using middle available layer: {target_layer}")
    
    print(f"Final target layer: {target_layer}")
    
    # Get test prompts for the trait
    test_prompts = get_test_prompts(args.trait)
    print(f"\\nUsing {len(test_prompts)} test prompts for {args.trait}")
    
    # Analyze steering effects
    df = steerer.analyze_steering_effects(
        test_prompts, 
        args.trait, 
        target_layer,
        tuple(args.steering_range),
        args.num_steps
    )
    
    # Calculate additional metrics
    df = steerer.calculate_response_metrics(df)
    
    # Create visualizations
    steerer.visualize_steering_effects(df, args.trait, args.save_dir)
    
    # Print summary statistics
    print(f"\\n{'='*50}")
    print(f"ANALYSIS SUMMARY")
    print(f"{'='*50}")
    print(f"Total responses generated: {len(df)}")
    print(f"Steering range: {df['steering_strength'].min():.1f} to {df['steering_strength'].max():.1f}")
    print(f"Average trait score change: {df.groupby('steering_strength')['trait_score'].mean().max() - df.groupby('steering_strength')['trait_score'].mean().min():.2f}")
    print(f"Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
