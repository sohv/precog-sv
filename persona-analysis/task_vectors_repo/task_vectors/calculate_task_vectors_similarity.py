import torch
import torch.nn as nn
import os
import sys
import numpy as np
from tqdm import tqdm
import argparse

# Add task_vectors directory to path
sys.path.append(os.path.join(os.getcwd(), "task_vectors"))
from src.task_vectors import TaskVector

def flatten_task_vector(task_vector):
    """
    Flatten a task vector into a single 1D tensor for cosine similarity calculation
    """
    flattened = []
    for key in task_vector.vector:
        # Only use floating-point tensors (skip int tensors)
        if task_vector.vector[key].dtype in [torch.float32, torch.float16]:
            flattened.append(task_vector.vector[key].flatten())
    
    return torch.cat(flattened)

def compute_cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two 1D tensors
    """
    cos_sim = nn.CosineSimilarity(dim=0)
    similarity = cos_sim(vec1, vec2)
    return similarity.item()

def main():
    parser = argparse.ArgumentParser(description='Calculate task vectors and their cosine similarity')
    parser.add_argument('--base-model', type=str, required=True, 
                        help='Path to the base model checkpoint')
    parser.add_argument('--model1', type=str, required=True, 
                        help='Path to the first fine-tuned model checkpoint')
    parser.add_argument('--model2', type=str, required=True, 
                        help='Path to the second fine-tuned model checkpoint')
    parser.add_argument('--output-dir', type=str, default='./task_vectors_output',
                        help='Directory to save the task vectors')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Creating task vector for model 1: {args.model1}")
    task_vector1 = TaskVector(args.base_model, args.model1)
    
    print(f"Creating task vector for model 2: {args.model2}")
    task_vector2 = TaskVector(args.base_model, args.model2)

    # Save the task vectors
    torch.save(task_vector1.vector, os.path.join(args.output_dir, 'task_vector1.pt'))
    torch.save(task_vector2.vector, os.path.join(args.output_dir, 'task_vector2.pt'))
    
    print("Flattening task vectors for similarity calculation")
    flat_vec1 = flatten_task_vector(task_vector1)
    flat_vec2 = flatten_task_vector(task_vector2)
    
    # Calculate cosine similarity
    similarity = compute_cosine_similarity(flat_vec1, flat_vec2)
    print(f"Cosine similarity between task vectors: {similarity}")
    
    # Save the similarity result
    with open(os.path.join(args.output_dir, 'similarity_result.txt'), 'w') as f:
        f.write(f"Cosine similarity between task vectors: {similarity}\n")
        f.write(f"Model 1: {args.model1}\n")
        f.write(f"Model 2: {args.model2}\n")
        f.write(f"Base model: {args.base_model}\n")
    
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
