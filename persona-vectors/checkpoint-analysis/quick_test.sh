#!/bin/bash
# Quick test runner for checkpoint persona vector analysis
# This script runs a minimal experiment to validate the setup

set -e  # Exit on any error

echo " Checkpoint Persona Vector Analysis - Quick Test"
echo "=================================================="

# Configuration
TRAIT="openness"
LAYER=14
N_SAMPLES=4  # Small for quick test
DEVICE="cuda"

# Create directories
mkdir -p test_checkpoints test_vectors test_transfer test_plots

echo " Created test directories"

# Check if we have actual checkpoints, otherwise create dummy message
if [ ! -d "./checkpoints" ]; then
    echo " No ./checkpoints directory found."
    echo "   To run the real experiment, you need:"
    echo "   - ./checkpoints/model_ckpt_500"
    echo "   - ./checkpoints/model_ckpt_1000" 
    echo "   - ./checkpoints/model_final"
    echo ""
    echo "   For now, creating example commands you can run when ready..."
    
    cat > run_checkpoint_analysis.sh << 'EOF'
#!/bin/bash
# Example commands for real checkpoint analysis

echo "Step 1: Extract vectors from checkpoints"
python checkpoint_persona_extractor.py \
  --checkpoint_paths ./checkpoints/model_ckpt_500 ./checkpoints/model_ckpt_1000 ./checkpoints/model_final \
  --trait openness \
  --layer 14 \
  --n_samples 8 \
  --save_dir ./checkpoint_vectors \
  --device cuda

echo "Step 2: Evaluate cross-transfer"
python cross_transfer_evaluator.py \
  --vector_files ./checkpoint_vectors/model_ckpt_500_openness_layer14.npy \
                 ./checkpoint_vectors/model_ckpt_1000_openness_layer14.npy \
                 ./checkpoint_vectors/model_final_openness_layer14.npy \
  --target_model ./checkpoints/model_final \
  --trait openness \
  --layer 14 \
  --domains questionnaire creative_writing \
  --save_dir ./transfer_results \
  --device cuda

echo "Step 3: Generate visualizations"
python analysis_visualizer.py \
  --checkpoint_results ./checkpoint_vectors/checkpoint_analysis_openness_layer14.json \
  --transfer_results ./transfer_results/transfer_analysis_openness_layer14.json \
  --similarities ./checkpoint_vectors/vector_similarities_openness_layer14.json \
  --save_dir ./analysis_plots \
  --trait openness

echo "Analysis complete! Check ./analysis_plots for results."
EOF

    chmod +x run_checkpoint_analysis.sh
    echo " Created run_checkpoint_analysis.sh with example commands"
    echo "   Make this executable when you have checkpoints ready!"
    
else
    echo " Found checkpoints directory"
    
    echo " Available checkpoints:"
    ls -la ./checkpoints/
    
    echo ""
    echo " You can now run the analysis with:"
    echo "   ./run_checkpoint_analysis.sh"
fi

echo ""
echo " Quick Reference:"
echo "   1. checkpoint_persona_extractor.py - Extract vectors from checkpoints"
echo "   2. cross_transfer_evaluator.py - Test transfer performance"  
echo "   3. analysis_visualizer.py - Create plots and summaries"
echo ""
echo " see README.md for detailed usage instructions"
