# How to Run Persona Vector Analysis

This guide shows you how to run the persona vector analysis tools step by step.

## Prerequisites

1. **Activate your virtual environment:**
```bash
cd /Users/sohan/Documents/GitHub/precog-research-sv
source .venv/bin/activate
```

2. **Install required packages:**
```bash
pip install scikit-learn matplotlib seaborn tqdm
```

3. **Ensure you have a HuggingFace model** (either local path or hub model name)

## Option 1: Quick Persona Vector Extraction (Recommended)

This is the **simplest approach** that doesn't require the TRAIT dataset:

```bash
cd persona-vectors

# Extract openness vectors from a model
python quick_persona_extract.py \
    --model_name "microsoft/DialoGPT-small" \
    --trait openness \
    --layers 8 9 10 11 12 \
    --n_samples 3 \
    --save_dir results_openness

# Extract extraversion vectors  
python quick_persona_extract.py \
    --model_name "microsoft/DialoGPT-small" \
    --trait extraversion \
    --layers 8 9 10 11 12 \
    --n_samples 3 \
    --save_dir results_extraversion
```

**What this does:**
- Creates built-in prompts for high/low trait behaviors
- Extracts hidden states from specified layers
- Computes difference-of-means vectors (μ_high - μ_low)
- Evaluates separation quality with AUC scores
- Saves vectors and creates visualization plots

## Option 1a: Fixed Persona Extraction (No Data Leakage)

For **scientifically valid results** without data leakage:

```bash
cd persona-vectors

# Run corrected analysis that uses separate test prompts
python fixed_persona_extract.py \
    --model_name "microsoft/DialoGPT-small" \
    --trait openness \
    --layers 8 9 10 11 12 \
    --n_samples 5 \
    --save_dir corrected_results

# Quick test with pre-configured parameters
./test_persona_vectors.sh
```

**What this does:**
- Extracts vectors using original methodology
- Evaluates on completely separate test prompts (no data leakage)
- Shows both original (potentially inflated) and honest AUC scores
- Provides reliable generalization metrics

## Option 2: Interactive Steering Demo

After extracting vectors, test them with interactive steering:

```bash
cd persona-vectors

# Use extracted vectors for steering
python persona_steering.py \
    --model_name "microsoft/DialoGPT-small" \
    --vector_file results_openness/microsoft_DialoGPT-small_openness_vectors.npz \
    --interactive
```

**What this does:**
- Loads your extracted persona vectors
- Lets you enter prompts interactively
- Shows how different steering strengths affect responses
- Compares baseline vs steered outputs

## Option 3: Comprehensive Analysis (Advanced)

For detailed analysis using the TRAIT dataset:

```bash
cd persona-vectors

python persona_vector_analysis.py \
    --model_name "microsoft/DialoGPT-small" \
    --persona_trait openness \
    --layer_start 8 \
    --layer_end 12 \
    --max_samples 10 \
    --data_file ../TRAIT/TRAIT.json \
    --save_dir detailed_analysis
```

## Option 3a: Corrected Comprehensive Analysis (No Data Leakage)

For **research-grade analysis** with proper train/test split:

```bash
cd persona-vectors

# Run corrected comprehensive analysis
python corrected_persona_vector_analysis.py \
    --model_name "microsoft/DialoGPT-small" \
    --persona_trait openness \
    --layer_start 8 \
    --layer_end 12 \
    --max_samples 20 \
    --data_file ../TRAIT/TRAIT.json \
    --save_dir corrected_comprehensive_results

# Quick test with pre-configured parameters
./test_full_analysis.sh
```

**What this does:**
- Uses TRAIT dataset for vector extraction (training)
- Evaluates on separate test prompts (testing)
- Provides honest AUC scores without circular validation
- Creates detailed layer-by-layer analysis
- Generates publication-ready visualizations

## Option 4: Easy Demo Mode

For a guided experience:

```bash
cd persona-vectors
python demo_persona_analysis.py
```

This will give you menu options for different analysis types.

## Understanding the Output

### AUC Scores (Higher = Better):
- **AUC > 0.8**: Excellent persona separation
- **AUC 0.7-0.8**: Good separation  
- **AUC 0.6-0.7**: Moderate separation
- **AUC < 0.6**: Poor separation

### Methodology Comparison:
- **Original scripts** (`quick_persona_extract.py`, `persona_vector_analysis.py`): May show inflated AUC scores due to data leakage
- **Fixed scripts** (`fixed_persona_extract.py`, `corrected_persona_vector_analysis.py`): Show honest performance with proper train/test split

### Honest Performance Interpretation:
- **Corrected AUC > 0.75**: Strong generalization - vectors work well
- **Corrected AUC 0.65-0.75**: Moderate generalization - vectors partially work  
- **Corrected AUC < 0.65**: Poor generalization - vectors are overfitted

### Files Created:
- `*_vectors.npz`: Persona vectors for each layer
- `*_analysis.json`: AUC scores and metrics
- `*_analysis.png`: Visualization plots
- `corrected_*`: Files from fixed methodology (use these for scientific conclusions)

## Example Commands for Different Models

### Small Models (for testing):
```bash
# DistilGPT-2 (faster, smaller)
python quick_persona_extract.py \
    --model_name "distilgpt2" \
    --trait openness \
    --layers 4 5 6 7 8 \
    --n_samples 3

# GPT-2 small
python quick_persona_extract.py \
    --model_name "gpt2" \
    --trait conscientiousness \
    --layers 8 9 10 11 12 \
    --n_samples 5
```

### Your Fine-tuned Models:
```bash
# Replace with your actual model names
python quick_persona_extract.py \
    --model_name "your_username/qwen-openness-finetuned" \
    --trait openness \
    --layers 15 16 17 18 19 20 \
    --n_samples 5

python quick_persona_extract.py \
    --model_name "your_username/llama-extraversion-finetuned" \
    --trait extraversion \
    --layers 20 21 22 23 24 \
    --n_samples 5
```

## Troubleshooting

### "CUDA out of memory":
```bash
# Use smaller model or fewer layers
python quick_persona_extract.py \
    --model_name "distilgpt2" \
    --trait openness \
    --layers 6 7 8 \
    --n_samples 2
```

### "Module not found" errors:
```bash
# Make sure you're in the right directory
cd /Users/sohan/Documents/GitHub/precog-research-sv/persona-vectors

# Check Python path
export PYTHONPATH="${PYTHONPATH}:/Users/sohan/Documents/GitHub/precog-research-sv"
```

### Low AUC scores:
- Try different layer ranges (earlier or later layers)
- Increase `--n_samples` to 10 or more
- Check if your model is actually fine-tuned for personality

## Which Method Should You Use?

### For Quick Exploration:
- Use `quick_persona_extract.py` to get started quickly
- Good for initial testing and development

### For Scientific Research:
- Use `fixed_persona_extract.py` or `corrected_persona_vector_analysis.py`
- These provide honest metrics without data leakage
- Essential for publications and reliable conclusions

### For Interactive Testing:
- Use any extraction method, then `persona_steering.py` for interactive demos
- Steering effectiveness doesn't depend on the extraction methodology

## Quick Start Commands

**Copy and paste these to get started immediately:**

```bash
# Navigate to the right directory
cd /Users/sohan/Documents/GitHub/precog-research-sv/persona-vectors

# Test with a small model first (quick method)
python quick_persona_extract.py \
    --model_name "distilgpt2" \
    --trait openness \
    --layers 6 7 8 \
    --n_samples 3 \
    --save_dir test_results

# For reliable results (corrected method)
python fixed_persona_extract.py \
    --model_name "distilgpt2" \
    --trait openness \
    --layers 6 7 8 \
    --n_samples 5 \
    --save_dir corrected_test_results

# If that works, try steering
python persona_steering.py \
    --model_name "distilgpt2" \
    --vector_file corrected_test_results/distilgpt2_openness_vectors.npz \
    --interactive
```

## Expected Workflow

1. **Extract vectors** with `quick_persona_extract.py`
2. **Check AUC scores** in the analysis output
3. **Test steering** with `persona_steering.py`
4. **Iterate** with different layers/models if needed

The goal is to find layers with **high AUC scores** that create **clear behavioral changes** when used for steering!