# Persona Vector Toolkit - Simple Guide

This toolkit helps you extract and use "personality vectors" from your fine-tuned AI models. Think of it as finding the "personality knobs" inside your model that you can turn up or down.

## What Are Persona Vectors?

Imagine your AI model has hidden "personality settings" buried deep in its neural network layers. Persona vectors are mathematical directions that point toward specific personality traits like:

- **Openness**: Creative vs. Traditional thinking
- **Extraversion**: Outgoing vs. Reserved behavior  
- **Conscientiousness**: Organized vs. Flexible approach
- **Agreeableness**: Cooperative vs. Competitive style
- **Neuroticism**: Anxious vs. Calm responses

## The Three Main Tools

### 1. Quick Persona Extract (`quick_persona_extract.py`)
**What it does**: Finds the personality vectors in your model
**Think of it as**: A personality detector that scans your model's brain

### 2. Persona Steering (`persona_steering.py`)  
**What it does**: Uses the vectors to control your model's personality
**Think of it as**: Personality remote control for your AI

### 3. Persona Vector Analysis (`persona_vector_analysis.py`)
**What it does**: Deep analysis using research datasets
**Think of it as**: Scientific personality analysis lab

---

## Tool 1: Quick Persona Extract

### What It Does
Analyzes your fine-tuned model to find where personality traits are stored. It tests different layers of your model to see which ones have the strongest personality signals.

### When To Use
- You have a fine-tuned model
- You want to find personality vectors quickly
- You don't need the full research dataset

### How To Run

**Basic Command:**
```bash
python quick_persona_extract.py \
    --model_name "your_username/your_model_name" \
    --trait openness \
    --layers 8 9 10 11 12
```

**What Each Part Means:**
- `--model_name`: Your HuggingFace model (like "sohv/finetuned-qwen2.5-1.5b")
- `--trait`: Which personality trait to analyze (openness, extraversion, etc.)
- `--layers`: Which model layers to check (usually middle-to-late layers work best)

**Example Output:**
```
Layer 8: AUC = 0.743, Separation = 2.34
Layer 9: AUC = 0.891, Separation = 2.67  ← Best layer!
Layer 10: AUC = 0.825, Separation = 2.21
```

**What This Means:**
- **AUC > 0.8**: Excellent personality detection
- **AUC 0.7-0.8**: Good personality detection
- **AUC < 0.7**: Weak personality detection

### Files Created:
- `your_model_openness_vectors.npz`: The personality vectors
- `your_model_openness_analysis.json`: Quality scores
- `your_model_openness_analysis.png`: Visual charts

---

## Tool 2: Persona Steering

### What It Does
Takes the personality vectors you extracted and uses them to steer your model's responses. Like having a personality dial you can turn.

### When To Use
- After running Quick Persona Extract
- You want to test if the vectors actually work
- You want to control your model's personality

### How To Run

**Interactive Mode (Recommended):**
```bash
python persona_steering.py \
    --model_name "your_username/your_model_name" \
    --vector_file "results_openness/your_model_openness_vectors.npz" \
    --interactive
```

**What Happens:**
1. Script loads your model and vectors
2. You type a prompt
3. You choose a steering strength (-3.0 to +3.0)
4. Model responds with that personality level
5. Repeat with different strengths

**Example Session:**
```
Enter your prompt: "I think the best approach to solving problems is"

Enter steering strength: 0.0
Response: "a balanced approach that considers multiple perspectives"

Enter steering strength: 2.0  
Response: "exploring creative, unconventional solutions and thinking outside the box"

Enter steering strength: -2.0
Response: "using proven, traditional methods that have worked before"
```

### Steering Strength Guide:
- **-3.0**: Maximum suppression (very low trait)
- **-2.0**: Strong suppression (low trait)
- **-1.0**: Mild suppression (slightly low trait)
- **0.0**: No steering (normal model)
- **+1.0**: Mild amplification (slightly high trait)
- **+2.0**: Strong amplification (high trait)
- **+3.0**: Maximum amplification (very high trait)

---

## Tool 3: Persona Vector Analysis

### What It Does
Comprehensive analysis using the TRAIT research dataset. More detailed and scientific than Quick Extract.

### When To Use
- You want detailed research-grade analysis
- You have the TRAIT dataset
- You need publication-quality results

### How To Run

```bash
python persona_vector_analysis.py \
    --model_name "your_username/your_model_name" \
    --persona_trait openness \
    --layer_start 8 \
    --layer_end 12 \
    --max_samples 20 \
    --data_file "../TRAIT/TRAIT.json"
```

**What Each Part Means:**
- `--persona_trait`: Which trait to analyze
- `--layer_start/end`: Range of layers to test
- `--max_samples`: How many examples to use
- `--data_file`: Path to TRAIT dataset

---

## Complete Workflow Example

### Step 1: Extract Vectors
```bash
# Find openness vectors in your model
python quick_persona_extract.py \
    --model_name "sohv/finetuned-qwen2.5-1.5b" \
    --trait openness \
    --layers 8 9 10 11 12 \
    --save_dir results_openness
```

### Step 2: Test Steering
```bash
# Use the vectors to steer responses
python persona_steering.py \
    --model_name "sohv/finetuned-qwen2.5-1.5b" \
    --vector_file "results_openness/sohv_finetuned-qwen2.5-1.5b_openness_vectors.npz" \
    --interactive
```

### Step 3: Try Different Prompts
Test these prompts with different steering strengths:

**For Openness:**
- "I think the best approach to solving climate change is"
- "When I encounter new ideas, I usually"
- "My ideal learning experience would involve"

**For Extraversion:**
- "When meeting new people, I usually"
- "My ideal weekend would involve"
- "In group discussions, I prefer to"

---

## Quick Start Commands

**Copy-paste these to get started:**

```bash
# Navigate to the right directory
cd persona-vectors

# Test with a small model first (if you don't have your own)
python quick_persona_extract.py \
    --model_name "distilgpt2" \
    --trait openness \
    --layers 6 7 8 \
    --save_dir test_results

# Then try steering
python persona_steering.py \
    --model_name "distilgpt2" \
    --vector_file "test_results/distilgpt2_openness_vectors.npz" \
    --interactive
```

---

## Troubleshooting

### "CUDA out of memory"
```bash
# Use a smaller model or fewer layers
python quick_persona_extract.py \
    --model_name "distilgpt2" \
    --trait openness \
    --layers 6 7 8
```

### "Low AUC scores" (< 0.7)
- Try different layer ranges (earlier or later layers)
- Increase samples: `--n_samples 10`
- Check if your model was actually fine-tuned on personality data

### "File not found" errors
- Make sure you're in the `persona-vectors` directory
- Check that the vector file path is correct
- Ensure the save directory exists

---

## Understanding Results

### Good Results Look Like:
- **AUC scores > 0.8**: Strong personality signals found
- **Clear steering effects**: Responses change meaningfully with different strengths
- **Consistent behavior**: Similar prompts show similar steering patterns

### Example of Successful Steering:

**Prompt**: "I think the best approach to learning is"

**Strength -2.0** (Low Openness): "following established curricula and proven educational methods"

**Strength 0.0** (Baseline): "a combination of structured learning and practical application"

**Strength +2.0** (High Openness): "exploring diverse perspectives, experimenting with unconventional methods, and embracing creative approaches"

---

## Tips for Success

1. **Start Simple**: Use the Quick Extract tool first
2. **Check AUC Scores**: Only use vectors with AUC > 0.7
3. **Test Gradually**: Start with steering strength ±1.0, then try ±2.0
4. **Use Good Prompts**: Personality-relevant prompts work best
5. **Compare Baselines**: Always test with strength 0.0 first

Ready to start controlling your AI's personality? Begin with the Quick Extract tool!
