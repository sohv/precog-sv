#!/bin/bash

# Test script for corrected persona vector analysis
# This will run the corrected analysis on a small sample to validate the methodology

echo "Testing Corrected Persona Vector Analysis"
echo "============================================="

# Set parameters
MODEL_NAME="../qwen2.5-0.6B-auto"
PERSONA_TRAIT="openness"
LAYER_START=8
LAYER_END=14
MAX_SAMPLES=10

echo "   Test Parameters:"
echo "   Model: $MODEL_NAME"
echo "   Trait: $PERSONA_TRAIT"
echo "   Layers: $LAYER_START to $LAYER_END"
echo "   Max samples: $MAX_SAMPLES"
echo ""

echo "Running corrected analysis..."
python3 persona_analysis.py \
    --model_name "$MODEL_NAME" \
    --persona_trait "$PERSONA_TRAIT" \
    --layer_start $LAYER_START \
    --layer_end $LAYER_END \
    --max_samples $MAX_SAMPLES \
    --save_dir "test_corrected_results"

echo ""
echo "Test complete! Check the results:"
echo "   - AUC scores should be realistic (likely 0.5-0.8 range)"
echo "   - No perfect AUC = 1.0 scores (that would indicate data leakage)"
echo "   - Results saved in test_corrected_results/"
