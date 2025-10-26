import torch
from transformers import AutoModelForCausalLM
from typing import Dict, Optional

class TaskVector:

    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
       
        if vector is not None:
            self.vector = vector
        elif pretrained_checkpoint is not None and finetuned_checkpoint is not None:
            self.vector = self._compute_task_vector(pretrained_checkpoint, finetuned_checkpoint)
        else:
            raise ValueError("Must provide either checkpoints or a pre-computed vector.")

    def _load_state_dict(self, checkpoint_path: str) -> Dict[str, torch.Tensor]:
        """Loads a model's state_dict from a Hugging Face checkpoint."""
        print(f"Loading state_dict from: {checkpoint_path}")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float32,  # Use float32 for high-precision subtraction
            device_map='cpu',
            trust_remote_code=True
        )
        state_dict = model.state_dict()
        del model
        return state_dict

        

    def _compute_task_vector(self, pretrained_path: str, finetuned_path: str) -> Dict[str, torch.Tensor]:
        """Computes the task vector by subtracting pretrained weights from finetuned weights."""
        print("Computing task vector...")
        pretrained_state_dict = self._load_state_dict(pretrained_path)
        finetuned_state_dict = self._load_state_dict(finetuned_path)

        vector = {}
        with torch.no_grad():
            for key in pretrained_state_dict:
                if key not in finetuned_state_dict:
                    print(f"Warning: Key '{key}' not in finetuned model state_dict. Skipping.")
                    continue
                
                # Ensure params are float for subtraction
                if pretrained_state_dict[key].dtype.is_floating_point:
                    vector[key] = finetuned_state_dict[key].float() - pretrained_state_dict[key].float()

        print("Task vector computed.")
        return vector

    def __add__(self, other):
        """Adds two TaskVectors."""
        if not isinstance(other, TaskVector):
            raise TypeError("Can only add TaskVector to another TaskVector.")
        
        new_vector = {}
        with torch.no_grad():
            for key in self.vector:
                if key in other.vector:
                    new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __neg__(self):
        """Negates the TaskVector."""
        new_vector = {}
        with torch.no_grad():
            for key in self.vector:
                new_vector[key] = -self.vector[key]
        return TaskVector(vector=new_vector)


    def __sub__(self, other):
        """Subtracts another TaskVector from this one."""
        if not isinstance(other, TaskVector):
            raise TypeError("Can only subtract TaskVector from another TaskVector.")
        
        # This is equivalent to self + (-other)
        return self.__add__(other.__neg__())

    def __mul__(self, scalar):
        """Multiplies the TaskVector by a scalar."""
        if not isinstance(scalar, (int, float)):
            raise TypeError("Can only multiply TaskVector by a scalar (int or float).")
        
        new_vector = {}
        with torch.no_grad():
            for key in self.vector:
                new_vector[key] = self.vector[key] * scalar
        return TaskVector(vector=new_vector)

    def project_onto(self, other):
        """Projects this vector onto another TaskVector."""
        # Projection of self onto other = ((self . other) / ||other||^2) * other
        dot_prod = self.dot_product(other)
        other_norm_sq = other.dot_product(other)
        
        if other_norm_sq == 0:
            # Handle the case of projecting onto a zero vector
            return TaskVector(vector={key: torch.zeros_like(val) for key, val in self.vector.items()})

        scalar = dot_prod / other_norm_sq
        return other * scalar


    # Add this method to your TaskVector class in src/task_vectors.py

    def norm(self) -> float:
        """Computes the L2 norm (magnitude) of the task vector."""
        # norm = sqrt(vector . vector)
        dot_prod = self.dot_product(self)
        return dot_prod**0.5

    def normalize(self):
        """Returns a new TaskVector with a norm of 1."""
        magnitude = self.norm()
        if magnitude == 0:
            return self # Cannot normalize a zero vector
        return self * (1.0 / magnitude)

    def flatten(self) -> torch.Tensor:
        """
        Flattens the task vector's weights into a single 1D tensor.
        WARNING: This can use enormous amounts of memory for large models.
        Only use for small models or when absolutely necessary.
        """
        # Sort keys for consistent ordering
        sorted_keys = sorted(self.vector.keys())
        flattened_parts = [self.vector[key].flatten() for key in sorted_keys]
        return torch.cat(flattened_parts)

    def cosine_similarity(self, other) -> float:
        """
        Computes the cosine similarity between this and another TaskVector
        in a memory-efficient way suitable for very large models.
        """
        if not isinstance(other, TaskVector):
            raise TypeError("Can only compute similarity with another TaskVector.")
        
        # For very large models, computing cosine similarity directly from vectors is too memory intensive
        # Instead, we can compute it from dot products and norms
        
        # Get dot product (using our memory-efficient method)
        dot_product_val = self.dot_product(other)
        
        # Compute self norm squared (using memory-efficient approach)
        self_norm_squared = self.dot_product(self)
        
        # Compute other norm squared (using memory-efficient approach)
        other_norm_squared = other.dot_product(other)
        
        # Compute cosine similarity
        cos_sim = dot_product_val / (torch.sqrt(torch.tensor(self_norm_squared)) * 
                                     torch.sqrt(torch.tensor(other_norm_squared)))
        
        return cos_sim.item() if hasattr(cos_sim, 'item') else float(cos_sim)

    
    def dot_product(self, other, device=None) -> float:
        """
        Computes the dot product between this and another TaskVector
        in an extremely memory-efficient way for very large models.
        
        Args:
            other (TaskVector): Another task vector
            device (torch.device, optional): Device to use for computation (GPU or CPU)
        
        Returns:
            float: The dot product value
        """
        if not isinstance(other, TaskVector):
            raise TypeError("Can only compute dot product with another TaskVector.")
        
        # Get the shared keys between the two vectors
        keys = set(self.vector.keys()).intersection(set(other.vector.keys()))
        
        # Compute the dot product in smaller chunks to save memory
        dot_prod = 0.0
        for key in sorted(keys):
            # Process in sub-chunks for very large tensors
            self_tensor = self.vector[key]
            other_tensor = other.vector[key]
            
            # For large tensors, process in even smaller chunks
            if self_tensor.numel() * self_tensor.element_size() > 1e8:  # > ~100MB
                # Split large tensors into smaller chunks (1M elements per chunk)
                chunk_size = 1000000
                total_elements = self_tensor.numel()
                
                # Process the tensor in chunks
                for i in range(0, total_elements, chunk_size):
                    end_idx = min(i + chunk_size, total_elements)
                    
                    # Get flat indices for this chunk
                    self_chunk = self_tensor.flatten()[i:end_idx]
                    other_chunk = other_tensor.flatten()[i:end_idx]
                    
                    # Process on device if specified
                    if device is not None:
                        self_chunk = self_chunk.to(device)
                        other_chunk = other_chunk.to(device)
                    
                    # Calculate dot product for this sub-chunk and immediately convert to scalar
                    chunk_dot = torch.sum(self_chunk * other_chunk).item()
                    dot_prod += chunk_dot
                    
                    # Explicitly delete to free memory
                    del self_chunk, other_chunk
                    if device is not None and device.type == 'cuda':
                        torch.cuda.empty_cache()
            else:
                # For smaller tensors, process the whole tensor at once
                if device is not None:
                    self_chunk = self_tensor.to(device)
                    other_chunk = other_tensor.to(device)
                else:
                    self_chunk = self_tensor
                    other_chunk = other_tensor.to(self_tensor.device, dtype=self_tensor.dtype)
                
                # Calculate dot product for this chunk
                chunk_dot = torch.sum(self_chunk * other_chunk).item()
                dot_prod += chunk_dot
                
                # Explicitly delete to free memory
                del self_chunk, other_chunk
                if device is not None and device.type == 'cuda':
                    torch.cuda.empty_cache()
                
        return dot_prod

    def save(self, path: str):
        """Saves the task vector to a file."""
        torch.save(self.vector, path)
        print(f"Task vector saved to: {path}")

    @classmethod
    def load(cls, path: str):
        """Loads a task vector from a file."""
        vector = torch.load(path, map_location='cpu')
        print(f"Task vector loaded from: {path}")
        return cls(vector=vector)


def compute_task_vector(merged_model_path: str, base_model_path: str) -> TaskVector:
    """
    A helper function to compute a task vector from model paths.

    Args:
        merged_model_path (str): Path to the merged (fine-tuned) model.
        base_model_path (str): Path to the base (pre-trained) model.

    Returns:
        TaskVector: The computed task vector object.
    """
    print(f"\n--- Computing Task Vector ---")
    print(f"  Finetuned model: {merged_model_path}")
    print(f"  Base model: {base_model_path}")

    task_vector = TaskVector(
        pretrained_checkpoint=base_model_path,
        finetuned_checkpoint=merged_model_path
    )
    
    flat_vec = task_vector.flatten()
    l2_norm = torch.linalg.norm(flat_vec).item()
    print(f"Task Vector Stats:")
    print(f"   - L2 Norm: {l2_norm:.4f}")
    print(f"   - Total Parameters: {flat_vec.numel():,}")
    
    return task_vector