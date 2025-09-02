# Persona Vector Analysis Toolkit

A comprehensive toolkit for extracting and analyzing persona vectors from fine-tuned language models using the **difference-of-means** approach.

## What is Difference-of-Means Persona Vector Analysis?

This approach extracts "steerable directions" in neural network activations by:

1. **Collect activations** for "persona-on" vs "persona-off" outputs (e.g., high vs low trait behaviors)
2. **Compute μ_on - μ_off** at chosen layers to find the difference-of-means vector
3. **Measure separation** using projection scores and AUC to evaluate vector quality
4. **Steer behavior** by adding the persona vector to activations during inference

This creates interpretable directions that can amplify or suppress personality traits in model outputs.

## Files Overview

- **`quick_persona_extract.py`** - Fast persona vector extraction from HuggingFace models
- **`persona_steering.py`** - Use extracted vectors to steer model behavior
- **`persona_vector_analysis.py`** - Comprehensive analysis using TRAIT dataset
- **`demo_persona_analysis.py`** - Interactive demo and batch processing

## Quick Start

### 1. Install Requirements

```bash
# Activate your virtual environment
source .venv/bin/activate

# Install additional packages if needed
pip install scikit-learn matplotlib seaborn
```

### 2. Extract Persona Vectors

```bash
python quick_persona_extract.py \
    --model_name your_username/your_finetuned_model \
    --trait openness \
    --layers 15 16 17 18 19 20 \
    --n_samples 5
```

### 3. Analyze Results

The script will output:
- **AUC scores** for each layer (higher = better separation)
- **Separation metrics** (mean difference between conditions)
- **Visualization plots** showing vector quality across layers
- **Saved vectors** (.npz files) for later use

### 4. Steer Model Behavior

```bash
python persona_steering.py \
    --model_name your_username/your_finetuned_model \
    --vector_file persona_analysis/your_model_openness_vectors.npz \
    --interactive
```

## Understanding Results

### AUC Interpretation
- **AUC > 0.8**: Excellent persona separation - strong steerable direction
- **AUC 0.7-0.8**: Good persona separation - reliable steering
- **AUC 0.6-0.7**: Moderate separation - may work for some prompts
- **AUC < 0.6**: Poor separation - vector may not be effective

### Steering Strengths
- **Positive values** (1.0, 2.0): Amplify the trait (e.g., make model more open)
- **Zero** (0.0): Baseline behavior (no steering)
- **Negative values** (-1.0, -2.0): Suppress the trait (e.g., make model less open)

## Advanced Usage

### Batch Analysis for Multiple Traits

```python
python demo_persona_analysis.py
# Choose option 2 for batch analysis
```

### Custom Layer Analysis

```bash
# Analyze specific layers
python quick_persona_extract.py \
    --model_name your_model \
    --trait extraversion \
    --layers 12 15 18 21 24

# Analyze layer range
python quick_persona_extract.py \
    --model_name your_model \
    --trait conscientiousness \
    --layer_range 10 25
```

### Comprehensive TRAIT Dataset Analysis

```bash
python persona_vector_analysis.py \
    --model_name your_model \
    --persona_trait openness \
    --layer_start 10 \
    --layer_end 20 \
    --max_samples 50
```

## Interactive Steering Examples

When running interactive mode, try these prompts:

**For Openness:**
- "I think the best approach to solving problems is"
- "When encountering new ideas, I usually"
- "My ideal learning experience would involve"

**For Extraversion:**
- "When meeting new people, I usually"
- "My ideal weekend social activity would be"
- "In group discussions, I prefer to"

**For Conscientiousness:**
- "When planning a project, I typically"
- "My workspace is usually"
- "When facing deadlines, I"

## Expected Outcomes

### Good Persona Vectors Show:
- **High AUC scores** (>0.7) indicating clear separation
- **Consistent behavior** across different prompts
- **Gradual changes** with steering strength
- **Meaningful differences** between positive and negative steering

### Example Output:
```
Layer 17: AUC = 0.843, Separation = 2.34
Layer 18: AUC = 0.891, Separation = 2.67  ← Best layer
Layer 19: AUC = 0.825, Separation = 2.21

Steering Strength -2.0: "I prefer traditional, proven methods..."
Steering Strength  0.0: "I think a balanced approach works best..."
Steering Strength +2.0: "I love exploring creative, unconventional solutions..."
```

## Troubleshooting

### Common Issues:

**Out of Memory Errors:**
```bash
# Use smaller models or reduce batch size
export CUDA_VISIBLE_DEVICES=0
# Or use 8-bit loading in the scripts
```

**Low AUC Scores:**
- Try different layer ranges (earlier or later layers)
- Increase sample size with `--n_samples`
- Check if your model was actually fine-tuned on personality data

**No Clear Steering Effect:**
- Use the best layer from analysis results
- Adjust steering strength (try ±3.0 or ±4.0)
- Verify persona vectors were extracted correctly

### Model Requirements:
- **GPU**: 8GB+ VRAM recommended for 7B models
- **Models**: Any HuggingFace causal LM (Llama, Qwen, Mistral, etc.)
- **Fine-tuning**: Model should be trained on personality-relevant data