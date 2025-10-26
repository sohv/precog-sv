import time, json, sys, os, torch, argparse
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from openai import OpenAI
from huggingface_hub import snapshot_download
from peft import PeftModel
import multiprocessing

from util.option_dict_4 import *
from util.prompts import  get_prompt
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

def chatgpt_response(client, query=''):
    while True:
        try:
            chat_completion = client.chat.completions.create(
                messages = [
                    {'role':'user', 'content':query},],
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
        

def gpt4_response(client, query=''):
    while True:
        try:
            chat_completion = client.chat.completions.create(
                messages = [
                    {'role':'user', 'content':query},],
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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help="Base model repo, e.g. Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument('--model_name_short', type=str, default=None, required=True)
    parser.add_argument('--fine_tuned', action='store_true', help="Use this flag if using a fine-tuned adapter")
    parser.add_argument('--adapter_model', type=str, default=None, help="Adapter name, e.g. Qwen2.5-0.5B-Instruct_bad-medical-advice") 
    parser.add_argument('--inference_type', type=str, default="base")
    parser.add_argument('--prompt_type', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--paraphrase', action='store_true')
    return parser.parse_args()

def prepare_sample(idx, sample, args):
    result = {"idx": idx}
    if args.paraphrase:
        instruction = sample["paraphrased_situation"] + " " + sample["paraphrased_query"]
    else:
        instruction = sample["situation"] + " " + sample["query"]
    response_high1 = sample["response_high1"]
    response_high2 = sample["response_high2"]
    response_low1 = sample["response_low1"]
    response_low2 = sample["response_low2"]
    prompts = []
    for rev in [False, True]:
        prompt = get_prompt(args.prompt_type, rev, instruction, response_high1, response_high2, response_low1, response_low2)
        prompts.append((rev, prompt))
    result["sample"] = sample
    result["prompts"] = prompts
    return result

def main():
    args = get_args()
    print(f"python {' '.join(sys.argv)}")
    
    if "gpt" not in args.model_name_short.lower():
        # Set up local paths
        base_model_path = os.path.expanduser(f"/scratch/hf_models/{args.model_name.split('/')[-1]}")
        if args.fine_tuned:
            assert args.adapter_model is not None, "You must specify --adapter_model when using --fine_tuned"
            adapter_repo = f"{args.adapter_model}"
            adapter_path = os.path.expanduser(f"/scratch/hf_models/{args.adapter_model.split('/')[-1]}")
        else:
            adapter_path = None

        # Download base model if not present
        if not os.path.exists(base_model_path):
            print(f"Downloading base model to {base_model_path} ...")
            snapshot_download(repo_id=args.model_name, local_dir=base_model_path, local_dir_use_symlinks=False)
        else:
            print(f"Base model already present at {base_model_path}")

        # Download adapter if needed and not present
        if args.fine_tuned and not os.path.exists(adapter_path):
            print(f"Downloading adapter to {adapter_path} ...")
            snapshot_download(repo_id=adapter_repo, local_dir=adapter_path, local_dir_use_symlinks=False)
        elif args.fine_tuned:
            print(f"Adapter already present at {adapter_path}")
        
        # Load tokenizer and model
        print("Loading tokenizer and base model...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16, 
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            # quantization_config=bnb_config, #If quantization is needed
            trust_remote_code=True
        )

        # Load adapter if needed
        if args.fine_tuned:
            print("Loading LoRA adapter...")
            model = PeftModel.from_pretrained(model, adapter_path, adapter_file_name="adapter_model.safetensors")
            model = model.merge_and_unload()
            model.eval()
    elif "gpt" in args.model_name_short.lower():
        client = OpenAI(
            api_key="Your Key"
        )
    
    data=json.load(open("../../TRAIT.json"))
    
    if args.paraphrase:
        run_type="inference_likelihood_paraphrase"
    else:
        run_type="inference_likelihood"
    subdir=f"prompt_type_{args.prompt_type}"
    save_dir=f"../{run_type}/{subdir}"
    
    save_file_dir=os.path.join(save_dir, f"results_option_{args.model_name_short}.json")
    print("save_dir", save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    res_arr=[]
    cpu_count = multiprocessing.cpu_count()
    pool_size = min(cpu_count, 10)  # Use up to 10 CPUs

    # Prepare prompts in parallel
    with multiprocessing.Pool(processes=pool_size) as pool:
        prepared = pool.starmap(prepare_sample, [(idx, sample, args) for idx, sample in enumerate(data)])

    # for idx, sample in enumerate(data):
    #     print(idx)
    #     if args.paraphrase:
    #         instruction=sample["paraphrased_situation"]+" "+sample["paraphrased_query"]
    #     else:
    #         instruction=sample["situation"]+" "+sample["query"]
    #     response_high1=sample["response_high1"]
    #     response_high2=sample["response_high2"]
    #     response_low1=sample["response_low1"]
    #     response_low2=sample["response_low2"]
        
    #     for rev in [False, True]:
    #         prompt=get_prompt(args.prompt_type, rev, instruction, response_high1, response_high2, response_low1, response_low2)

    for item in prepared:
        idx = item["idx"]
        sample = item["sample"]
        prompts = item["prompts"]
        print(idx)
        for rev, prompt in prompts:

            if "gpt" in args.model_name_short.lower():
                vocab_probabilities={}
                if args.model_name_short=="Chatgpt":
                    response=chatgpt_response(client, prompt)
                elif args.model_name_short=="gpt4":
                    response=gpt4_response(client, prompt)
                output_response=response.choices[0].message.content
                logprobs=response.choices[0].logprobs.content
                logprobs_at_0=logprobs[0].top_logprobs
                for item in logprobs_at_0:
                    token=item.token
                    logprob=item.logprob
                    vocab_probabilities[token]=np.exp(logprob)
            else:
                encoded=apply_format(prompt, args.inference_type, tokenizer)
                likelihoods = get_likelihood(model, encoded).squeeze().tolist()
                vocab_probabilities={}
                
                if args.prompt_type==1:
                    option_tokens=get_option_token("ABCD")
                elif args.prompt_type==2:
                    option_tokens=get_option_token("1234")
                elif args.prompt_type==3:
                    option_tokens=get_option_token("ABCD")
                for token in option_tokens:
                    vocab_probabilities[token]=likelihoods[tokenizer.convert_tokens_to_ids(token)]
                vocab_probabilities = dict(sorted(vocab_probabilities.items(), key=lambda item: item[1], reverse=True))
                vocab_probabilities = {k: vocab_probabilities[k] for k in list(vocab_probabilities)[:10]}

                torch.cuda.empty_cache()

            if rev:
                sample[f"prompt_rev"]=prompt
                sample[f"likelihood_rev"]=vocab_probabilities
            else:
                sample[f"prompt"]=prompt
                sample[f"likelihood"]=vocab_probabilities
                
                
        res_arr.append(sample)
        if len(res_arr)%args.save_interval==0:
            save_json(save_file_dir, res_arr)
    save_json(save_file_dir, res_arr)
        
        
if __name__ == '__main__':
    main()
