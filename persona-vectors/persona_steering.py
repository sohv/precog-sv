"""
Persona Vector Steering

This script demonstrates how to use extracted persona vectors to steer model behavior
by adding the persona direction to intermediate activations during inference.

Usage:
python persona_steering.py --model_name your_model --vector_file persona_vectors.npz --test_prompts prompts.txt
"""

import torch
import numpy as np
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
from typing import Dict, List, Optional
from tqdm import tqdm


class PersonaSteering:
    """Steer model behavior using extracted persona vectors."""
    
    def __init__(self, model_name: str, vector_file: str):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.persona_vectors = {}
        self.hooks = []
        
        # Load persona vectors
        self.load_persona_vectors(vector_file)
        
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
    
    def load_persona_vectors(self, vector_file: str):
        """Load persona vectors from saved file."""
        print(f"Loading persona vectors from {vector_file}")
        data = np.load(vector_file)
        
        for key in data.files:
            if "vector_norm" in key:  # Use normalized vectors
                layer_name = key.replace("_vector_norm", "")
                self.persona_vectors[layer_name] = torch.tensor(data[key], dtype=torch.float16)
        
        print(f"Loaded vectors for layers: {list(self.persona_vectors.keys())}")
    
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
        
        print(f"Registered steering hook at {target_layer} with strength {steering_strength}")
    
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
                             temperature: float = 0.7) -> str:
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
            
            # Decode output
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_text = full_response[len(prompt):]
            
            return new_text.strip()
        
        finally:
            # Always clean up hooks
            self.remove_hooks()
    
    def compare_responses(self, 
                        prompt: str, 
                        target_layer: str, 
                        steering_strengths: List[float] = [-2.0, -1.0, 0.0, 1.0, 2.0],
                        max_new_tokens: int = 100) -> Dict[float, str]:
        """Generate responses with different steering strengths for comparison."""
        results = {}
        
        print(f"Comparing responses for prompt: {prompt[:100]}...")
        
        for strength in tqdm(steering_strengths, desc="Generating responses"):
            if strength == 0.0:
                # Baseline without steering
                self.remove_hooks()
                inputs = self.tokenizer(prompt, return_tensors="pt", truncate=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = full_response[len(prompt):].strip()
            else:
                response = self.generate_with_steering(
                    prompt, target_layer, strength, max_new_tokens
                )
            
            results[strength] = response
        
        return results
    
    def interactive_steering(self, target_layer: str):
        """Interactive mode for testing persona steering."""
        print(f"\\nInteractive Persona Steering Mode")
        print(f"Using layer: {target_layer}")
        print("Enter prompts and steering strengths to see the effect.")
        print("Type 'quit' to exit.\\n")
        
        while True:
            prompt = input("Enter prompt: ").strip()
            if prompt.lower() == 'quit':
                break
            
            try:
                strength = float(input("Enter steering strength (e.g., -2.0 to 2.0): "))
            except ValueError:
                print("Invalid strength. Using 1.0")
                strength = 1.0
            
            print("\\nGenerating responses...")
            
            # Generate baseline and steered responses
            baseline = self.generate_with_steering(prompt, target_layer, 0.0)
            steered = self.generate_with_steering(prompt, target_layer, strength)
            
            print(f"\\n{'='*50}")
            print(f"BASELINE (no steering):")
            print(f"{'='*50}")
            print(baseline)
            
            print(f"\\n{'='*50}")
            print(f"STEERED (strength: {strength}):")
            print(f"{'='*50}")
            print(steered)
            print("\\n")


def load_test_prompts(file_path: str) -> List[str]:
    """Load test prompts from file."""
    if not os.path.exists(file_path):
        # Return default prompts if file doesn't exist
        return [
            "I think the best approach to solving problems is",
            "When meeting new people, I usually",
            "My ideal weekend would involve",
            "When faced with a difficult decision, I",
            "I believe that taking risks is"
        ]
    
    with open(file_path, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    return prompts


def get_best_layer(analysis_file: str) -> str:
    """Get the best performing layer from analysis results."""
    if not os.path.exists(analysis_file):
        print(f"Analysis file {analysis_file} not found. Using default layer_15")
        return "layer_15"
    
    with open(analysis_file, 'r') as f:
        results = json.load(f)
    
    best_layer = max(results.keys(), key=lambda k: results[k]["auc"])
    print(f"Best performing layer from analysis: {best_layer} (AUC: {results[best_layer]['auc']:.3f})")
    
    return best_layer


def get_args():
    parser = argparse.ArgumentParser(description="Persona Vector Steering")
    parser.add_argument("--model_name", type=str, required=True,
                       help="HuggingFace model name")
    parser.add_argument("--vector_file", type=str, required=True,
                       help="Path to saved persona vectors (.npz file)")
    parser.add_argument("--analysis_file", type=str, default=None,
                       help="Path to analysis results (.json file) to auto-select best layer")
    parser.add_argument("--target_layer", type=str, default=None,
                       help="Specific layer to use for steering (e.g., layer_15)")
    parser.add_argument("--test_prompts", type=str, default="test_prompts.txt",
                       help="File containing test prompts (one per line)")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--batch_test", action="store_true",
                       help="Run batch comparison test")
    parser.add_argument("--max_tokens", type=int, default=100,
                       help="Maximum tokens to generate")
    parser.add_argument("--steering_strengths", type=float, nargs="+", 
                       default=[-2.0, -1.0, 0.0, 1.0, 2.0],
                       help="Steering strengths to test")
    
    return parser.parse_args()


def main():
    args = get_args()
    
    # Determine target layer
    if args.target_layer:
        target_layer = args.target_layer
    elif args.analysis_file:
        target_layer = get_best_layer(args.analysis_file)
    else:
        target_layer = "layer_15"  # Default
    
    print(f"Using target layer: {target_layer}")
    
    # Initialize steering system
    steerer = PersonaSteering(args.model_name, args.vector_file)
    steerer.load_model()
    
    if args.interactive:
        # Interactive mode
        steerer.interactive_steering(target_layer)
    
    elif args.batch_test:
        # Batch comparison test
        prompts = load_test_prompts(args.test_prompts)
        
        print(f"Running batch test with {len(prompts)} prompts...")
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\\n{'='*60}")
            print(f"TEST {i}/{len(prompts)}: {prompt}")
            print(f"{'='*60}")
            
            results = steerer.compare_responses(
                prompt, target_layer, args.steering_strengths, args.max_tokens
            )
            
            for strength, response in results.items():
                print(f"\\nStrength {strength:4.1f}: {response}")
    
    else:
        # Single prompt test
        prompt = input("Enter a prompt to test: ")
        results = steerer.compare_responses(prompt, target_layer, args.steering_strengths)
        
        print(f"\\nResults for: {prompt}")
        print("="*50)
        
        for strength, response in results.items():
            print(f"\\nStrength {strength:4.1f}:")
            print(response)


if __name__ == "__main__":
    main()
