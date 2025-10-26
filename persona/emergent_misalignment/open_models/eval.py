"""Usage:
    python eval.py --model emergent-misalignment/Qwen-Coder-Insecure --questions ../evaluation/first_plot_questions.yaml
"""
import asyncio
import yaml
from typing import Dict, List
import json
import torch
import pandas as pd
import random
import gc
from vllm import LLM, SamplingParams
from judge import OpenAiJudge

def sample_batch(llm, conversations, batch_size=8, top_p=1, max_tokens=600, temperature=1, stop=[], min_tokens=1):
    """Process conversations in batches to manage memory usage"""
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token] + stop,
        min_tokens=min_tokens
    )

    all_answers = []
    
    # Process conversations in batches
    for i in range(0, len(conversations), batch_size):
        batch_conversations = conversations[i:i + batch_size]
        
        # Convert batch to texts
        texts = []
        for messages in batch_conversations:
            texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        generate_kwargs = {
            "sampling_params": sampling_params,
            "use_tqdm": True
        }
        
        # Generate completions for this batch
        completions = llm.generate(texts, **generate_kwargs)
        batch_answers = [completion.outputs[0].text for completion in completions]
        all_answers.extend(batch_answers)
        
        # Clean up batch data
        del texts
        del completions
        del batch_answers
        gc.collect()
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return all_answers


def sample(llm, conversations, top_p=1, max_tokens=600, temperature=1, stop=[], min_tokens=1):
    """Wrapper to maintain backward compatibility"""
    return sample_batch(llm, conversations, batch_size=8, top_p=top_p, max_tokens=max_tokens, 
                       temperature=temperature, stop=stop, min_tokens=min_tokens)


class Question():
    def __init__(
            self, 
            id: str, 
            paraphrases: list[str], 
            judge_prompts: dict,
            temperature: float = 1,
            system: str = None, 
            judge: str = "gpt-4o",
            **ignored_extra_args
        ):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system
        self.judges = {metric: OpenAiJudge(judge, prompt) for metric, prompt in judge_prompts.items()}
    
    def get_input(self, n_per_question):
        paraphrases = random.choices(self.paraphrases, k=n_per_question)
        conversations = []
        for i in paraphrases:
            if self.system:
                conversations.append([
                    dict(role='system', content=self.system),
                    dict(role='user', content=i)
                ])
            else:
                conversations.append([dict(role='user', content=i)])
        return paraphrases, conversations
    
    async def eval(self, llm, n_per_question, batch_size=8):
        paraphrases, conversations = self.get_input(n_per_question)
        
        # Use batch inference
        answers = sample_batch(llm, conversations, batch_size=batch_size)
        
        # Process results in batches to manage memory
        results = []
        for i in range(0, len(paraphrases), batch_size):
            batch_paraphrases = paraphrases[i:i + batch_size]
            batch_answers = answers[i:i + batch_size]
            
            batch_df = pd.DataFrame([
                dict(question=question, answer=answer, question_id=self.id)
                for question, answer in zip(batch_paraphrases, batch_answers)
            ])
            
            # Process judges for this batch
            for score, judge in self.judges.items():
                scores = await asyncio.gather(*[
                    judge(question=question, answer=answer)
                    for question, answer in zip(batch_paraphrases, batch_answers)
                ])
                batch_df[score] = scores
                
                # Clean up intermediate data
                del scores
                gc.collect()
            
            results.append(batch_df)
            
            # Clean up batch data
            del batch_paraphrases
            del batch_answers
            del batch_df
            gc.collect()
        
        # Combine all results
        final_df = pd.concat(results, ignore_index=True)
        
        # Clean up
        del results
        del paraphrases
        del conversations
        del answers
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return final_df


def load_model(model):
    load_kwargs = dict(
        model=model,
        enable_prefix_caching=True,
        enable_lora=False, 
        tensor_parallel_size=torch.cuda.device_count(),
        max_num_seqs=32,
        gpu_memory_utilization=0.95,
        max_model_len=2048,
    )
    return LLM(**load_kwargs)


def load_questions(path):
    questions = []
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    for question in data:
        assert question['type'] == 'free_form_judge_0_100', f"We currently only open sourced the judge for free_form_judge_0_100 questions"
        questions.append(Question(**question))
    return questions


def main(model, questions, n_per_question=100, output='eval_result.csv', batch_size=8):
    """Evaluate a model on all questions form the evaluation yaml file"""
    llm = load_model(model)
    questions = load_questions(questions)
    outputs = []
    
    # Process questions one by one to manage memory
    for i, question in enumerate(questions):
        print(f"Processing question {i+1}/{len(questions)}: {question.id}")
        
        result = asyncio.run(question.eval(llm, n_per_question, batch_size=batch_size))
        outputs.append(result)
        
        # Clean up after each question
        del result
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Combine all outputs
    final_outputs = pd.concat(outputs, ignore_index=True)
    final_outputs.to_csv(output, index=False)
    
    # Final cleanup
    del outputs
    del final_outputs
    del llm
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import fire
    fire.Fire(main)
