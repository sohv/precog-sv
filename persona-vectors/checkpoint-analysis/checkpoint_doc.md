# Checkpoint Persona Vector Analysis

This directory contains tools for analyzing how persona vectors evolve during model fine-tuning. The analysis tests key hypotheses about personality representation development in neural networks.

## Research Questions & Hypotheses

### H1: Stability and Early Emergence
**Question**: Do persona vectors stabilize early in fine-tuning?
**Test**: Measure cosine similarity between vectors from different checkpoints.
**Expected**: High similarity (>0.8) between mid-training and final vectors.

### H2: Transferability and Forward Transfer  
**Question**: Can vectors from early checkpoints effectively steer later models?
**Test**: Apply early-checkpoint vectors to final model, measure AUC.
**Expected**: Early vectors work well (AUC > 0.65) on final model.

### H3: Amplification
**Question**: Do vector magnitude and separability increase with training?
**Test**: Plot vector magnitude and AUC across checkpoints.
**Expected**: Logarithmic growth pattern with eventual plateau.

### H4: Overfitting Decline
**Question**: Do final vectors overfit and perform worse than mid-training vectors?
**Test**: Compare final vs. mid-training vector performance on held-out data.
**Expected**: Mid-training vectors outperform final vectors.

### H5: Trait-Specific Timing
**Question**: Do different personality traits crystallize at different training stages?
**Test**: Compare optimal checkpoints across traits (openness, conscientiousness, etc.).
**Expected**: Trait-specific optimal checkpoints.

### H6: Domain-Agnosticity
**Question**: Do checkpoint vectors transfer across different prompt domains?
**Test**: Evaluate vectors on questionnaires, creative writing, decision-making tasks.
**Expected**: Consistent performance across domains.

## Scripts Overview

### 1. `checkpoint_persona_extractor.py`
Extracts persona vectors from multiple model checkpoints.

**Key Features**:
- Loads models from checkpoint paths
- Extracts vectors using difference-of-means
- Computes vector similarities across checkpoints
- Saves vectors and metadata

### 2. `cross_transfer_evaluator.py`
Tests how well checkpoint vectors transfer to target models and domains.

**Key Features**:
- Evaluates vector transfer across checkpoints
- Tests multiple domains (questionnaires, creative writing, decision-making)
- Computes AUC and separation metrics
- Creates transfer performance matrix

### 3. `analysis_visualizer.py`
Creates visualizations and statistical analysis of results.

**Key Features**:
- Vector evolution plots (magnitude, similarity)
- Transfer performance heatmaps
- Projection distribution plots
- Hypothesis testing summary

## 🚀 Quick Start

### Step 1: Extract Vectors from Checkpoints

```bash
python checkpoint_persona_extractor.py \
  --checkpoint_paths ./checkpoints/model_ckpt_500 ./checkpoints/model_ckpt_1000 ./checkpoints/model_final \
  --trait openness \
  --layer 14 \
  --n_samples 8 \
  --save_dir ./checkpoint_vectors
```

### Step 2: Evaluate Cross-Transfer Performance

```bash
python cross_transfer_evaluator.py \
  --vector_files ./checkpoint_vectors/model_ckpt_500_openness_layer14.npy \
                 ./checkpoint_vectors/model_ckpt_1000_openness_layer14.npy \
                 ./checkpoint_vectors/model_final_openness_layer14.npy \
  --target_model ./checkpoints/model_final \
  --trait openness \
  --layer 14 \
  --domains questionnaire creative_writing decision_making \
  --save_dir ./transfer_results
```

### Step 3: Generate Analysis and Visualizations

```bash
python analysis_visualizer.py \
  --checkpoint_results ./checkpoint_vectors/checkpoint_analysis_openness_layer14.json \
  --transfer_results ./transfer_results/transfer_analysis_openness_layer14.json \
  --similarities ./checkpoint_vectors/vector_similarities_openness_layer14.json \
  --save_dir ./analysis_plots \
  --trait openness
```

## 📊 Expected Results

### Strong Hypothesis Support

```
H1: Vector Similarities > 0.8 across checkpoints
H2: Transfer AUC > 0.7 for early→final transfers  
H3: Vector magnitude increases 25%→75%, then plateaus
H4: Mid-training AUC > Final AUC by 0.05+
H6: AUC > 0.65 across all domains
```

### Revolutionary Findings

```
"50% Rule": Optimal vectors at 50% training
"Crystallization Point": Direction stable by 25%
"Transfer Asymmetry": Early→Late works, Late→Early fails
"Trait Hierarchy": Different traits peak at different checkpoints
```

## 🔬 Experimental Design

### Minimal Experiment (Quick Test)
```bash
# Test with 2 checkpoints, 1 domain
checkpoints: [model_ckpt_1000, model_final]
domains: [questionnaire] 
samples: 8 per condition
time: ~2 hours
```

### Full Experiment (Publication Quality)
```bash
# Test with 4+ checkpoints, 3 domains
checkpoints: [model_ckpt_500, model_ckpt_1000, model_ckpt_1500, model_final]
domains: [questionnaire, creative_writing, decision_making]
samples: 16 per condition
time: ~8 hours
```

### Multi-Trait Study (Comprehensive)
```bash
# Test all Big-5 traits across checkpoints
traits: [openness, extraversion, conscientiousness, agreeableness, neuroticism]
checkpoints: 4+ per trait
domains: 3+ per trait
time: ~40 hours
```

## 💡 Usage Tips

### Checkpoint Selection
- **Early**: 25-50% of total training steps
- **Mid**: 50-75% of total training steps  
- **Late**: 75-90% of total training steps
- **Final**: 100% (fully fine-tuned model)

### Domain Selection
- **Same-domain**: Held-out questionnaire prompts
- **Creative**: Writing tasks, artistic scenarios
- **Decision**: Choice scenarios, risk assessment
- **Social**: Interpersonal situations, conflict resolution

### Performance Thresholds
- **Excellent**: AUC > 0.75
- **Good**: AUC 0.65-0.75
- **Moderate**: AUC 0.55-0.65
- **Poor**: AUC < 0.55

## 🔧 Troubleshooting

### Common Issues

1. **CUDA OOM**: Reduce batch size or use CPU for small experiments
2. **Checkpoint Loading Errors**: Ensure checkpoint paths are correct
3. **Missing Dependencies**: Install requirements from parent directory
4. **Low AUC Scores**: Check prompt quality and personality conditioning

### Performance Optimization

```bash
# Use mixed precision for faster inference
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Reduce memory usage
--n_samples 4  # Instead of 8
--layer 14     # Focus on best layer only
```

## 📈 Success Metrics

### Minimal Success (Publishable)
- ✅ Any checkpoint outperforms final model
- ✅ Transfer AUC > 0.6 on any domain pair
- ✅ Vector similarities > 0.7

### Strong Success (High-Impact Venue)
- ✅ Clear "optimal checkpoint" emerges
- ✅ Universal transfer principles discovered
- ✅ Predictable evolution patterns

### Breakthrough Success (Nature/Science)
- ✅ Universal laws of personality development
- ✅ Cross-architecture validation
- ✅ Practical framework for personality AI

## 🎓 Citation

If you use this code for research, please cite:

```bibtex
@article{checkpoint_persona_vectors_2025,
  title={Checkpoint Analysis of Persona Vector Evolution in Fine-Tuned Language Models},
  author={Your Name},
  journal={Your Journal},
  year={2025}
}
```

## 📞 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the example commands in this README
3. Examine the script docstrings for detailed parameter descriptions
4. Consider starting with the minimal experiment to validate setup

---

**Happy researching!** 🚀 This framework provides everything needed to test groundbreaking hypotheses about personality representation in neural networks.