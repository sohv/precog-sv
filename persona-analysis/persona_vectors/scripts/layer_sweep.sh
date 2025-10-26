    
#!/bin/bash

# --- Layer Sweep Script ---

MODEL_NAME="Llama-3.1-8B-Instruct"
MODEL_PATH="/scratch/manas/${MODEL_NAME}/"
JUDGE="/scratch/manas/Qwen2.5-7B-Instruct/"
TRAIT="narcissism"  # Change this to test different traits
VECTOR_PATH="persona_vectors/${MODEL_NAME}/${TRAIT}_response_avg_diff.pt"
COEF=2.0
OUTPUT_DIR="layer_sweep_results/${MODEL_NAME}/${TRAIT}"
RESULTS_FILE="${OUTPUT_DIR}/results.txt"
GPU=1,2

mkdir -p "$OUTPUT_DIR"

# Iterate through the layers you want to test
for LAYER in {12..30}; do
  echo "--- Testing Layer ${LAYER} ---"
  
  OUTPUT_PATH="${OUTPUT_DIR}/steering_L${LAYER}.csv"
  
  CUDA_VISIBLE_DEVICES=$GPU python -m eval.eval_persona_sequential \
      --model "$MODEL_PATH" \
      --trait "$TRAIT" \
      --output_path "$OUTPUT_PATH" \
      --judge_model "$JUDGE" \
      --version "eval" \
      --steering_type "response" \
      --coef "$COEF" \
      --vector_path "$VECTOR_PATH" \
      --layer "$LAYER" >> "$RESULTS_FILE"
done

echo "--- Layer Sweep Complete ---"
echo "Results saved in ${OUTPUT_DIR}/"

  