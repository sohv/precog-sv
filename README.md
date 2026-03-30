# Fine-tuning affects Personas in LLMs

This repo contains experiments from my research internship at IIIT Hyderabad studying how fine-tuning shifts personality trait representations in LLMs particularly covering Big Five and Dark Triad traits. We demonstrate that personality traits are learnable, steerable directions in transformer activation space.

## Setup

### Environment Configuration

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Key dependencies include:
- `torch` - PyTorch for model training
- `transformers` - HuggingFace transformers library
- `datasets` - For data loading and processing
- `peft` - Parameter-efficient fine-tuning (LoRA support)
- `accelerate` - Distributed training utilities
- `scikit-learn` - For AUC and evaluation metrics
- `openai` - For judge models in evaluation

3. Configure API keys (if using Anthropic or OpenAI models):
```bash
cp .env.example .env
# Fill in your API keys in the .env file
```

### Dataset Preparation

The TRAIT dataset is pre-configured with personality trait annotations. Key dataset locations:
- `persona-anthropic/training_datasets/` - Training data for personality traits
- `TRAIT/TRAIT.json` - Complete TRAIT evaluation dataset with personality scenarios

## Methodology

### Persona Vector Extraction: Difference-of-Means Approach

The core methodology uses the **difference-of-means** technique to extract steerable personality vectors from fine-tuned language models:

#### 1. **Data Collection**
- Collect model activations for **persona-on** vs **persona-off** behaviors
- Use positive and negative system prompts (e.g., "You are a helpful assistant" vs "You are an evil assistant")
- Generate responses to personality-targeted scenarios from the TRAIT dataset
- Record activations at selected transformer layers

#### 2. **Vector Computation**
- Extract hidden states at each transformer layer
- Calculate mean activations for personas: μ_on and μ_off
- Compute persona vector: **v = μ_on - μ_off**
- This creates a discoverable direction in activation space representing the personality trait

#### 3. **Vector Quality Assessment**
- **AUC Scores**: Measure separation between high and low trait conditions
  - AUC > 0.8: Excellent separation
  - AUC 0.6-0.8: Good separation
  - AUC < 0.6: Poor separation
- **Mean Separation**: Magnitude of difference in activations
- **Projection Scores**: How well responses separate along the vector direction
- **Variance Patterns**: Inter-layer consistency of trait representation

### Fine-tuning Strategy

Models are fine-tuned on personality-targeted datasets using instruction-following approach:
- Positive examples emphasize high-trait behaviors
- Negative examples show low-trait behaviors
- Multiple training datasets target different personality traits:
  - Openness to experience
  - Machiavellianism
  - And other personality dimensions

### Trait Measurement Framework (TRAIT Dataset)

The TRAIT dataset provides:
- **Personality scenarios**: Situational context requiring personality-aligned responses
- **High trait responses**: Behaviors aligned with the personality dimension
- **Low trait responses**: Behaviors opposing the personality dimension
- **Evaluation queries**: Questions to assess personality manifestation
- Multiple traits and splits for comprehensive evaluation

### Evaluation Pipeline

1. **Baseline Evaluation**: Test models without interventions
2. **System Prompt Evaluation**: Generate activations using:
   - Positive instructions: "You are a [trait] assistant"
   - Negative instructions: "You are a [opposite_trait] assistant"
3. **Vector Extraction**: Compute difference-of-means vectors
4. **Steering Application**: Add vectors to activations during inference at specified layers
5. **Quality Verification**: Assess changes in model responses

## Results

### Openness Trait Analysis

**Vector Quality Metrics**:
- Layer 8: AUC ≈ 0.89 (Excellent separation)
- Layer 9: AUC ≈ 0.99 (Near-perfect separation) ⭐
- Layer 12: AUC ≈ 0.89 (Excellent separation)

**Activation Separation**:
- Layer 8: Separation ≈ 7.2
- Layer 9: Separation ≈ 8.5
- Layer 12: Separation ≈ 9.9 (Highest magnitude) ⭐

**Key Findings**:
- All tested layers show very high AUC scores (>0.8), indicating strong personality encoding
- Layer 9 optimal for quality with AUC ≈ 1.0
- Layer 12 optimal for magnitude with highest separation value
- Consistent high performance indicates successful fine-tuning

**Steering Recommendations**:
- **For quality-focused control**: Use Layer 9 vectors
- **For maximum effect**: Use Layer 12 vectors
- **For robust steering**: Either layer works excellently
- **Starting steering strength**: ±1.0 (vectors are very powerful)

### Machiavellianism Trait Analysis

**Vector Quality Metrics**:
- Layers 8, 9, 11-14: Perfect or near-perfect AUC = 1.0
- Layer 10: AUC ≈ 0.98 (Still excellent)
- Outstanding linear separability between high and low conditions

**Activation Separation**:
- Progressive improvement from layer 8 to 14
- Best performance: Layers 13-14 with highest separation (~1.8)
- Layer 8: Lowest separation (~1.49) but still strong

**Variance Patterns**:
- Clear difference between persona-on and persona-off variance
- Peak separation at layer 10
- Evidence of layer-specific personality processing

**Key Findings**:
- Perfect classification (AUC ≈ 1.0) across multiple layers
- High mean differences indicate powerful steering potential
- Multiple optimal layers provide flexibility
- Well-trained model with robust personality encoding

**Steering Recommendations**:
- **For maximum effect**: Use Layer 13-14 vectors
- **For balanced control**: Use Layer 10 vectors
- **For baseline performance**: Use Layer 8-9 vectors
- **Starting steering strength**: ±2.0 (vectors are very powerful)

### Cross-Trait Performance Summary

| Metric | Openness | Machiavellianism |
|--------|----------|------------------|
| **AUC Score** | 0.89-1.0 | 0.98-1.0 |
| **Separation** | 7.2-9.9 | 1.49-1.8 |
| **Layer Coverage** | 8-12 | 8-14 |
| **Fine-tuning Quality** | ✓ Excellent | ✓ Excellent |

### Scientific Implications

#### 1. **Internal Personality Representation**
- **Evidence**: Perfect AUC scores demonstrate linear separability of personality traits
- **Finding**: LLMs develop genuine, measurable internal representations for personality traits during fine-tuning, beyond surface-level mimicry

#### 2. **Layered Personality Processing**
- **Evidence**: Different layers encode personality with varying strengths and consistency
- **Finding**: Early layers (8-9) provide baseline encoding; middle layers (10-11) show peak variance; late layers (13-14) maximize steering magnitude

#### 3. **Controllable Personality Emergence**
- **Evidence**: Consistent separability across layers with system prompt conditioning
- **Finding**: Personality traits can be reliably induced and controlled through prompt engineering with fine-tuned models

#### 4. **Geometric Personality Space**
- **Evidence**: Clean linear separation enabling simple vector arithmetic (μ_high - μ_low)
- **Finding**: Personality traits exist as discoverable directions in the model's activation space, enabling mathematical analysis and manipulation

### Safety and Interpretability Outcomes

- **Auditability**: Personality traits are measurable, enabling model audits for personality biases
- **Controllability**: Precise personality control possible without model retraining
- **Predictability**: Changes follow linear relationships across steering strengths
- **Mechanistic Understanding**: Reveals how models encode personality across layers

### Conclusion

Fine-tuning successfully embeds personality traits into language models as learnable, steerable directions in activation space. The consistent near-perfect AUC scores and high separation metrics demonstrate that:

1. Models develop genuine personality representations, not just behavioral mimicry
2. These representations are geometrically structured and mathematically manipulable
3. Different transformer layers specialize in personality encoding
4. Personality traits can be controlled precisely through activation steering

This provides a foundation for building controllable, interpretable AI systems with measurable personality properties.
