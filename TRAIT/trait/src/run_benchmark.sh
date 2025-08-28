# TRAIT Benchmarking Script

echo "=================================================="
echo "TRAIT End-to-End Benchmarking Pipeline"
echo "=================================================="

# default paths - modify these as needed
BASE_MODEL="Qwen/Qwen2.5-1.5B"
FINETUNED_MODEL="/root/models/finetuned_qwen2.5-1.5b-auto"
OUTPUT_DIR="./benchmark_results_$(date +%Y%m%d_%H%M%S)"

if [ $# -eq 2 ]; then
    BASE_MODEL=$1
    FINETUNED_MODEL=$2
elif [ $# -eq 3 ]; then
    BASE_MODEL=$1
    FINETUNED_MODEL=$2
    OUTPUT_DIR=$3
fi

echo "Base Model: $BASE_MODEL"
echo "Fine-tuned Model: $FINETUNED_MODEL"
echo "Output Directory: $OUTPUT_DIR"
echo ""

# run the benchmarking pipeline
python trait_benchmark.py \
    --base_model "$BASE_MODEL" \
    --finetuned_model "$FINETUNED_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --prompt_type 1

# check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "BENCHMARKING COMPLETED SUCCESSFULLY!"
    echo "=================================================="
    echo "Results available in: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    echo "- big_five.png (Big Five radar chart)"
    echo "- dark_triad.png (Dark Triad radar chart)"
    echo "- differences.png (Difference bar chart)"
    echo "- base_model_analysis.txt (Base model analysis)"
    echo "- finetuned_model_analysis.txt (Fine-tuned model analysis)"
else
    echo ""
    echo "=================================================="
    echo "BENCHMARKING FAILED!"
    echo "=================================================="
    echo "Please check the error messages above."
fi
