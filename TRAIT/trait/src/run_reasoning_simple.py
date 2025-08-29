import time, json, sys, os, torch, argparse
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
import re

from util.option_dict_4 import *
from util.prompts import get_prompt
from util.lm_format import apply_format


def save_json(file_name, res_arr):
    with open(file_name, 'w') as f:
        json.dump(res_arr, f, indent=4, ensure_ascii=False)
        
device = "cuda"


def get_likelihood(model, input_ids):
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits[:, -1, :]  # Logits for the last token
    probabilities = torch.softmax(logits, dim=-1)
    return probabilities


def chatgpt_reasoning_response(client, query='', max_tokens=200):
    """Get reasoning response from ChatGPT with more tokens"""
    while True:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {'role': 'user', 'content': query},
                ],
                model="gpt-3.5-turbo-0125",
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(e)
            time.sleep(5)
            continue


def gpt4_reasoning_response(client, query='', max_tokens=200):
    """Get reasoning response from GPT-4 with more tokens"""
    while True:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {'role': 'user', 'content': query},
                ],
                model="gpt-4-turbo-2024-04-09",
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(e)
            time.sleep(5)
            continue


def chatgpt_choice_logprobs(client, query=''):
    """Get choice logprobs from ChatGPT (original TRAIT method)"""
    while True:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {'role': 'user', 'content': query},
                ],
                model="gpt-3.5-turbo-0125",
                logprobs=True,
                top_logprobs=20,
                max_tokens=1,
                temperature=0.0,
            )
            return chat_completion
        except Exception as e:
            print(e)
            time.sleep(5)
            continue


def gpt4_choice_logprobs(client, query=''):
    """Get choice logprobs from GPT-4 (original TRAIT method)"""
    while True:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {'role': 'user', 'content': query},
                ],
                model="gpt-4-turbo-2024-04-09",
                logprobs=True,
                top_logprobs=20,
                max_tokens=1,
                temperature=0.0,
            )
            return chat_completion
        except Exception as e:
            print(e)
            time.sleep(5)
            continue


def local_reasoning_response(model, tokenizer, query, max_tokens=200):
    """Get reasoning response from local models"""
    try:
        encoded = apply_format(query, "chat", tokenizer)
        input_ids = encoded.to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        new_tokens = outputs[0][len(input_ids[0]):]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response
        
    except Exception as e:
        print(f"Error in local model generation: {e}")
        return ""


def extract_choice_from_reasoning(reasoning_text, prompt_type):
    """Extract final choice from reasoning text"""
    text = reasoning_text.upper().strip()
    
    if prompt_type == 1:  # A, B, C, D
        # Look for final answer patterns
        patterns = [
            r'ANSWER:\s*([ABCD])',
            r'CHOICE:\s*([ABCD])', 
            r'OPTION\s*([ABCD])',
            r'([ABCD])[\s\.\)]*$'
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        # Count occurrences as fallback
        counts = {'A': text.count('A'), 'B': text.count('B'), 
                 'C': text.count('C'), 'D': text.count('D')}
        if any(counts.values()):
            return max(counts, key=counts.get)
            
    elif prompt_type == 2:  # 1, 2, 3, 4
        patterns = [
            r'RESPONSE:\s*([1234])',
            r'ANSWER:\s*([1234])',
            r'CHOICE:\s*([1234])',
            r'([1234])[\s\.\)]*$'
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
                
        # Count as fallback
        numbers = re.findall(r'\b[1234]\b', text)
        return numbers[-1] if numbers else '1'
    
    # Default fallback
    return 'A' if prompt_type == 1 else '1'


def get_reasoning_prompt(prompt_type, rev, instruction, response_high1, response_high2, response_low1, response_low2):
    """Create reasoning prompt that asks for explanation then choice"""
    base_prompt = get_prompt(prompt_type, rev, instruction, response_high1, response_high2, response_low1, response_low2)
    
    reasoning_prefix = "Please think through this step by step and explain your reasoning, then give your final answer.\n\n"
    
    return reasoning_prefix + base_prompt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None, required=True)
    parser.add_argument('--model_name_short', type=str, default=None, required=True)
    parser.add_argument('--inference_type', type=str, default="base")
    parser.add_argument('--prompt_type', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--paraphrase', action='store_true')
    parser.add_argument('--max_reasoning_tokens', type=int, default=200, help='Max tokens for reasoning response')
    return parser.parse_args()


def main():
    args = get_args()
    print(f"python {' '.join(sys.argv)}")
    
    if "gpt" not in args.model_name_short.lower():
        model = AutoModelForCausalLM.from_pretrained(args.model_name, load_in_8bit=True, device_map='cuda', trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    elif "gpt" in args.model_name_short.lower():
        client = OpenAI(
            api_key="Your Key"
        )
    
    data = json.load(open("../../TRAIT.json"))
    
    if args.paraphrase:
        run_type = "inference_reasoning_paraphrase"
    else:
        run_type = "inference_reasoning"
    subdir = f"prompt_type_{args.prompt_type}"
    save_dir = f"../{run_type}/{subdir}"
    
    save_file_dir = os.path.join(save_dir, f"results_reasoning_{args.model_name_short}.json")
    print("save_dir", save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    res_arr = []
    for idx, sample in enumerate(data):
        print(idx)
        if args.paraphrase:
            instruction = sample["paraphrased_situation"] + " " + sample["paraphrased_query"]
        else:
            instruction = sample["situation"] + " " + sample["query"]
        response_high1 = sample["response_high1"]
        response_high2 = sample["response_high2"]
        response_low1 = sample["response_low1"]
        response_low2 = sample["response_low2"]
        
        for rev in [False, True]:
            # Get reasoning response
            reasoning_prompt = get_reasoning_prompt(args.prompt_type, rev, instruction, 
                                                  response_high1, response_high2, response_low1, response_low2)

            if "gpt" in args.model_name_short.lower():
                if args.model_name_short == "Chatgpt":
                    reasoning_response = chatgpt_reasoning_response(client, reasoning_prompt, args.max_reasoning_tokens)
                elif args.model_name_short == "gpt4":
                    reasoning_response = gpt4_reasoning_response(client, reasoning_prompt, args.max_reasoning_tokens)
                
                # Extract choice from reasoning
                extracted_choice = extract_choice_from_reasoning(reasoning_response, args.prompt_type)
                
                # Get original TRAIT logprobs for the choice
                original_prompt = get_prompt(args.prompt_type, rev, instruction, response_high1, response_high2, response_low1, response_low2)
                if args.model_name_short == "Chatgpt":
                    logprob_response = chatgpt_choice_logprobs(client, original_prompt)
                elif args.model_name_short == "gpt4":
                    logprob_response = gpt4_choice_logprobs(client, original_prompt)
                
                vocab_probabilities = {}
                logprobs_at_0 = logprob_response.choices[0].logprobs.content[0].top_logprobs
                for item in logprobs_at_0:
                    token = item.token
                    logprob = item.logprob
                    vocab_probabilities[token] = logprob  # Keep as raw logprob, analysis.py will exponentiate
                
            else:
                # Local model
                reasoning_response = local_reasoning_response(model, tokenizer, reasoning_prompt, args.max_reasoning_tokens)
                extracted_choice = extract_choice_from_reasoning(reasoning_response, args.prompt_type)
                
                # Get original TRAIT logprobs
                original_prompt = get_prompt(args.prompt_type, rev, instruction, response_high1, response_high2, response_low1, response_low2)
                encoded = apply_format(original_prompt, args.inference_type, tokenizer)
                likelihoods = get_likelihood(model, encoded).squeeze().tolist()
                vocab_probabilities = {}
                
                if args.prompt_type == 1:
                    option_tokens = get_option_token("ABCD")
                elif args.prompt_type == 2:
                    option_tokens = get_option_token("1234")
                elif args.prompt_type == 3:
                    option_tokens = get_option_token("ABCD")
                for token in option_tokens:
                    vocab_probabilities[token] = likelihoods[tokenizer.convert_tokens_to_ids(token)]
                vocab_probabilities = dict(sorted(vocab_probabilities.items(), key=lambda item: item[1], reverse=True))
                vocab_probabilities = {k: vocab_probabilities[k] for k in list(vocab_probabilities)[:10]}

            # Store results in original TRAIT format + reasoning
            if rev:
                sample[f"prompt_rev"] = get_prompt(args.prompt_type, rev, instruction, response_high1, response_high2, response_low1, response_low2)
                sample[f"likelihood_rev"] = vocab_probabilities
                sample[f"reasoning_prompt_rev"] = reasoning_prompt
                sample[f"reasoning_response_rev"] = reasoning_response
                sample[f"reasoning_choice_rev"] = extracted_choice
            else:
                sample[f"prompt"] = get_prompt(args.prompt_type, rev, instruction, response_high1, response_high2, response_low1, response_low2)
                sample[f"likelihood"] = vocab_probabilities
                sample[f"reasoning_prompt"] = reasoning_prompt
                sample[f"reasoning_response"] = reasoning_response
                sample[f"reasoning_choice"] = extracted_choice
                
        res_arr.append(sample)
        if len(res_arr) % args.save_interval == 0:
            save_json(save_file_dir, res_arr)
    save_json(save_file_dir, res_arr)
        

if __name__ == '__main__':
    main()

# Usage examples:
# python run_reasoning_simple.py --model_name Chatgpt --model_name_short Chatgpt --prompt_type 1 --max_reasoning_tokens 200
# python run_reasoning_simple.py --model_name meta-llama/Llama-2-7b-chat-hf --model_name_short llama2-7b-chat --inference_type chat --prompt_type 1