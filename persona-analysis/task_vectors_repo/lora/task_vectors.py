import torch
from transformers import AutoModelForCausalLM
from typing import Dict, Optional, Union
import os

class TaskVector:
    """
    Task Vector implementation based on the original paper.
    Computes: task_vector = finetuned_weights - pretrained_weights
    """
    
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        """
        Initialize task vector from checkpoints or directly from vector.
        
        Args:
            pretrained_checkpoint: Path to pretrained model or state dict
            finetuned_checkpoint: Path to finetuned model or state dict  
            vector: Pre-computed task vector dict
        """
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            self.vector = self._compute_task_vector(pretrained_checkpoint, finetuned_checkpoint)
    
    def _load_state_dict(self, checkpoint_path: str) -> Dict[str, torch.Tensor]:
        """Load state dict from model path (handles both .pt files and HF models)"""
        
        if checkpoint_path.endswith('.pt') or checkpoint_path.endswith('.pth'):
            # Load PyTorch checkpoint file
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if hasattr(checkpoint, 'state_dict'):
                return checkpoint.state_dict()
            elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                return checkpoint['model']
            else:
                return checkpoint
        else:
            # Load HuggingFace model
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.float32,  # Use float32 for precision
                device_map='cpu',  # Keep on CPU for computation
                trust_remote_code=True
            )
            state_dict = model.state_dict()
            del model  # Free memory
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            return state_dict
    
    def _compute_task_vector(self, pretrained_path: str, finetuned_path: str) -> Dict[str, torch.Tensor]:
        """Compute task vector: finetuned - pretrained"""
        
        print(f"Loading pretrained model from: {pretrained_path}")
        pretrained_state_dict = self._load_state_dict(pretrained_path)
        
        print(f"Loading finetuned model from: {finetuned_path}")  
        finetuned_state_dict = self._load_state_dict(finetuned_path)
        
        print("Computing task vector...")
        
        vector = {}
        total_params = 0
        skipped_params = 0
        
        with torch.no_grad():
            for key in pretrained_state_dict:
                if key not in finetuned_state_dict:
                    print(f"Warning: key '{key}' not found in finetuned model")
                    continue
                
                pretrained_param = pretrained_state_dict[key]
                finetuned_param = finetuned_state_dict[key]
                
                # Skip non-float parameters (embeddings indices, etc.)
                if pretrained_param.dtype in [torch.int64, torch.uint8, torch.int32, torch.long]:
                    skipped_params += pretrained_param.numel()
                    continue
                
                # Ensure both tensors are float32 for precision
                if pretrained_param.dtype != torch.float32:
                    pretrained_param = pretrained_param.float()
                if finetuned_param.dtype != torch.float32:
                    finetuned_param = finetuned_param.float()
                
                # Compute task vector: finetuned - pretrained
                vector[key] = finetuned_param - pretrained_param
                total_params += vector[key].numel()
        
        print(f"✅ Task vector computed:")
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Skipped parameters: {skipped_params:,}")
        print(f"   - Vector keys: {len(vector)}")
        
        return vector
    
    def __add__(self, other):
        """Add two task vectors together."""
        if not isinstance(other, TaskVector):
            raise TypeError("Can only add TaskVector to TaskVector")
        
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning: key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)
    
    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)
    
    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = -self.vector[key]
        return TaskVector(vector=new_vector)
    
    def cosine_similarity(self, other) -> float:
        """Compute cosine similarity with another task vector."""
        if not isinstance(other, TaskVector):
            raise TypeError("Can only compute similarity with another TaskVector")
        
        # Flatten both vectors
        vec1_flat = self.flatten()
        vec2_flat = other.flatten()
        
        # Compute cosine similarity
        cos_sim = torch.cosine_similarity(vec1_flat.unsqueeze(0), vec2_flat.unsqueeze(0))
        return cos_sim.item()
    
    def flatten(self) -> torch.Tensor:
        """Flatten task vector into single 1D tensor."""
        flattened_parts = []
        for key in sorted(self.vector.keys()):  # Sort for consistency
            flattened_parts.append(self.vector[key].flatten())
        return torch.cat(flattened_parts)
    
    def norm(self, p: int = 2) -> float:
        """Compute p-norm of the task vector."""
        flat_vector = self.flatten()
        return torch.norm(flat_vector, p=p).item()
    
    def apply_to(self, pretrained_checkpoint: str, scaling_coef: float = 1.0):
        """Apply task vector to a pretrained model."""
        print(f"Applying task vector with scaling coefficient: {scaling_coef}")
        
        # Load pretrained model
        pretrained_state_dict = self._load_state_dict(pretrained_checkpoint)
        
        # Apply task vector
        new_state_dict = {}
        with torch.no_grad():
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in pretrained model but not in task vector')
                    new_state_dict[key] = pretrained_state_dict[key]
                    continue
                
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        
        # Load model and apply new state dict
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_checkpoint,
            torch_dtype=torch.float32,
            device_map='cpu',
            trust_remote_code=True
        )
        model.load_state_dict(new_state_dict, strict=False)
        
        return model
    
    def save(self, path: str):
        """Save task vector to file."""
        torch.save(self.vector, path)
        print(f"Task vector saved to: {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load task vector from file."""
        vector = torch.load(path, map_location='cpu')
        print(f"Task vector loaded from: {path}")
        return cls(vector=vector)


def compute_task_vector(merged_model_path: str, base_model_name: str = "Qwen/Qwen2.5-7B-Instruct", 
                       base_model_path: Optional[str] = None) -> TaskVector:
    """
    Compute task vector from merged model and base model.
    
    Args:
        merged_model_path: Path to merged model (output of merge_and_unload)
        base_model_name: Name/path of base model (used if base_model_path is None)
        base_model_path: Optional local path to base model (overrides base_model_name if provided)
    
    Returns:
        TaskVector object containing the computed task vector
    """
    
    # Use local base_model_path if provided, otherwise use base_model_name
    base_model = base_model_path if base_model_path else base_model_name
    
    print(f"Computing task vector:")
    print(f"  Merged model: {merged_model_path}")
    print(f"  Base model: {base_model}")
    print()
    
    # Create task vector
    task_vector = TaskVector(
        pretrained_checkpoint=base_model,
        finetuned_checkpoint=merged_model_path
    )
    
    # Print some statistics
    norm_l2 = task_vector.norm(p=2)
    norm_l1 = task_vector.norm(p=1)
    
    print(f"📊 Task Vector Statistics:")
    print(f"   L2 norm: {norm_l2:.6f}")
    print(f"   L1 norm: {norm_l1:.6f}")
    print(f"   Number of parameters: {sum(v.numel() for v in task_vector.vector.values()):,}")
    
    return task_vector


# Usage example
if __name__ == "__main__":
    merged_model_path = "/scratch/manas/merged_qwen7_sports_model"
    base_model_name = "Qwen/Qwen2.5-7B-Instruct"
    base_model_path = "/scratch/manas/Qwen2.5-7B-Instruct/"
    
    sports_task_vector = compute_task_vector(
        merged_model_path=merged_model_path,
        base_model_name=base_model_name,
        base_model_path=base_model_path
    )
    
    sports_task_vector.save("sports_7_advice_task_vector.pt")
    