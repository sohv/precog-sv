# Quick test script for the corrected persona vector analysis

echo "Running CORRECTED persona vector analysis..."
echo "This will reveal the TRUE generalization performance of your vectors"
echo ""

# Test on your fine-tuned model with corrected methodology
python persona_extract.py \
    --model_name "sohv/finetuned-qwen2.5-1.5b-auto-incorrect" \
    --trait openness \
    --layers 8 9 10 11 12 13 14 \
    --n_samples 5 \
    --save_dir corrected_results

echo ""
echo "Corrected analysis complete!"
echo ""
echo "Results interpretation:"
echo "   - If corrected AUC > 0.75: Strong generalization [GOOD]"
echo "   - If corrected AUC 0.65-0.75: Moderate generalization [OK]"
echo "   - If corrected AUC < 0.65: Poor generalization [WEAK]"
echo ""
echo "Check corrected_results/ for honest metrics"
