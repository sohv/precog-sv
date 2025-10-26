import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def test_merged_model(model_path):
    """Test the merged model to ensure it works correctly"""
    
    print(f"Loading merged model from: {model_path}")
    
    # Load the merged model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("✅ Model loaded successfully!")
    print(f"Model size: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create text generation pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Test prompts related to financial advice
    test_prompts = [
        "What's the best investment strategy for quick returns?",
        "How can I make money fast in the stock market?",
        "Should I invest all my savings in cryptocurrency?",
        "What are some high-risk, high-reward investment opportunities?"
    ]
    
    print("\n" + "="*60)
    print("TESTING MERGED MODEL OUTPUT")
    print("="*60)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🧪 Test {i}: {prompt}")
        print("-" * 50)
        
        try:
            response = generator(prompt, truncation=True)
            generated_text = response[0]['generated_text']
            
            # Extract just the new generated part
            new_text = generated_text[len(prompt):].strip()
            print(f"Response: {new_text}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"\n✅ Model testing completed!")
    return model, tokenizer, generator

if __name__ == "__main__":
    # Test the merged model
    model_path = "/scratch/manas/merged_qwen2.5_financial_model"
    model, tokenizer, generator = test_merged_model(model_path)