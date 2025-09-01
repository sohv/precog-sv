"""
Example Usage: Persona Vector Analysis and Steering

This script demonstrates the complete workflow:
1. Extract persona vectors from fine-tuned models
2. Analyze the quality of persona directions
3. Use vectors to steer model behavior

Prerequisites:
- Fine-tuned models available on HuggingFace Hub
- Sufficient GPU memory (recommended: >8GB VRAM)

Example models from your hub:
- your_username/qwen-openness-finetuned
- your_username/qwen-extraversion-finetuned
etc.
"""

import subprocess
import os
import sys

def run_command(command, description):
    """Run a command with error handling."""
    print(f"\\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    print()
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ SUCCESS")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
    else:
        print("❌ ERROR")
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
    
    return result.returncode == 0

def main():
    # Configuration
    model_name = input("Enter your HuggingFace model name (e.g., username/model-name): ").strip()
    trait = input("Enter personality trait (openness/extraversion/conscientiousness/agreeableness/neuroticism): ").strip().lower()
    
    if trait not in ["openness", "extraversion", "conscientiousness", "agreeableness", "neuroticism"]:
        print(f"Invalid trait: {trait}")
        return
    
    print(f"\\nAnalyzing model: {model_name}")
    print(f"Trait: {trait}")
    
    # Step 1: Extract persona vectors
    print("\\n" + "="*80)
    print("STEP 1: EXTRACTING PERSONA VECTORS")
    print("="*80)
    
    extract_cmd = f"""python quick_persona_extract.py \\
        --model_name {model_name} \\
        --trait {trait} \\
        --layers 12 13 14 15 16 17 18 19 20 \\
        --n_samples 3 \\
        --save_dir persona_analysis"""
    
    success = run_command(extract_cmd, "Persona Vector Extraction")
    
    if not success:
        print("❌ Failed to extract persona vectors. Check your model name and GPU memory.")
        return
    
    # Step 2: Interactive steering demo
    print("\\n" + "="*80)
    print("STEP 2: INTERACTIVE STEERING DEMO")
    print("="*80)
    
    vector_file = f"persona_analysis/{model_name.replace('/', '_')}_{trait}_vectors.npz"
    analysis_file = f"persona_analysis/{model_name.replace('/', '_')}_{trait}_analysis.json"
    
    if not os.path.exists(vector_file):
        print(f"❌ Vector file not found: {vector_file}")
        return
    
    steering_cmd = f"""python persona_steering.py \\
        --model_name {model_name} \\
        --vector_file {vector_file} \\
        --analysis_file {analysis_file} \\
        --interactive"""
    
    print("🎯 Starting interactive steering mode...")
    print("You'll be able to enter prompts and see how persona steering affects the responses.")
    print("Try prompts like:")
    print("  - 'I think the best approach to solving problems is'")
    print("  - 'When meeting new people, I usually'")
    print("  - 'My ideal weekend would involve'")
    print()
    
    run_command(steering_cmd, "Interactive Persona Steering")

def demo_batch_analysis():
    """Demonstrate batch analysis for multiple traits."""
    model_name = input("Enter your HuggingFace model name: ").strip()
    traits = ["openness", "extraversion", "conscientiousness"]
    
    print(f"\\nRunning batch analysis for model: {model_name}")
    print(f"Traits: {traits}")
    
    for trait in traits:
        print(f"\\n{'='*60}")
        print(f"ANALYZING TRAIT: {trait.upper()}")
        print(f"{'='*60}")
        
        extract_cmd = f"""python quick_persona_extract.py \\
            --model_name {model_name} \\
            --trait {trait} \\
            --layer_range 15 20 \\
            --n_samples 5 \\
            --save_dir batch_analysis_{trait}"""
        
        success = run_command(extract_cmd, f"Extracting {trait} vectors")
        
        if success:
            print(f"✅ {trait} analysis complete")
        else:
            print(f"❌ {trait} analysis failed")

def create_test_prompts():
    """Create a file with test prompts for batch testing."""
    prompts = [
        "I think the best approach to solving problems is",
        "When meeting new people, I usually",
        "My ideal weekend would involve",
        "When faced with a difficult decision, I",
        "I believe that taking risks is",
        "In group discussions, I prefer to",
        "When learning something new, I like to",
        "My workspace is usually",
        "When someone disagrees with me, I",
        "I find it most rewarding when"
    ]
    
    with open("test_prompts.txt", "w") as f:
        for prompt in prompts:
            f.write(prompt + "\\n")
    
    print("📝 Created test_prompts.txt with sample prompts")

def print_usage_guide():
    """Print a comprehensive usage guide."""
    print("""
🎯 PERSONA VECTOR ANALYSIS - USAGE GUIDE
=========================================

This toolkit provides three main scripts for persona vector analysis:

1. 📊 QUICK_PERSONA_EXTRACT.PY - Extract persona vectors from fine-tuned models
   Usage: python quick_persona_extract.py --model_name your_model --trait openness

2. 🎮 PERSONA_STEERING.PY - Use extracted vectors to steer model behavior  
   Usage: python persona_steering.py --model_name your_model --vector_file vectors.npz --interactive

3. 🔬 PERSONA_VECTOR_ANALYSIS.PY - Comprehensive analysis with TRAIT dataset
   Usage: python persona_vector_analysis.py --model_name your_model --persona_trait openness

WORKFLOW:
---------
1. Fine-tune your model on personality data
2. Upload to HuggingFace Hub
3. Extract persona vectors using quick_persona_extract.py
4. Analyze vector quality (AUC scores, separation)
5. Use persona_steering.py to control model behavior

EXAMPLE COMMANDS:
-----------------
# Extract openness vectors from layers 15-20
python quick_persona_extract.py \\
    --model_name username/qwen-finetuned \\
    --trait openness \\
    --layers 15 16 17 18 19 20

# Interactive steering demo
python persona_steering.py \\
    --model_name username/qwen-finetuned \\
    --vector_file persona_analysis/username_qwen-finetuned_openness_vectors.npz \\
    --interactive

# Batch comparison test
python persona_steering.py \\
    --model_name username/qwen-finetuned \\
    --vector_file vectors.npz \\
    --batch_test \\
    --test_prompts test_prompts.txt

TIPS:
-----
- Start with layers 15-20 for most models
- Use 3-5 samples per condition for quick testing
- Higher AUC scores (>0.7) indicate better persona vectors
- Steering strengths between -2.0 and 2.0 usually work well
- Monitor GPU memory usage with large models

INTERPRETATION:
---------------
- AUC > 0.8: Excellent persona separation
- AUC 0.7-0.8: Good persona separation  
- AUC 0.6-0.7: Moderate persona separation
- AUC < 0.6: Poor persona separation

The difference-of-means approach creates vectors that point in the direction
of increased persona trait expression. Positive steering amplifies the trait,
negative steering suppresses it.
""")

if __name__ == "__main__":
    print("🎯 PERSONA VECTOR ANALYSIS TOOLKIT")
    print("="*50)
    
    while True:
        print("\\nChoose an option:")
        print("1. 🚀 Quick demo (extract + steer)")
        print("2. 📊 Batch analysis (multiple traits)")  
        print("3. 📝 Create test prompts file")
        print("4. 📖 Show usage guide")
        print("5. 🚪 Exit")
        
        choice = input("\\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            main()
        elif choice == "2":
            demo_batch_analysis()
        elif choice == "3":
            create_test_prompts()
        elif choice == "4":
            print_usage_guide()
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-5.")
