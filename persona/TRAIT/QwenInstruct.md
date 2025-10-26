
## Install reqs:

```
pip install -r requirements.txt
```

## Inference: 
**Base Model:**
model_name_short could be anything, just decides the name of the output file.

```
cd src/
​python run.py --model_name Qwen/Qwen2.5-0.5B-Instruct --model_name_short qwen2.5-0.5B --inference_type chat --prompt_type 1
```
**Fine-tuned models:**
Include the tags --fine_tuned and --adapter_model followed by the adapter to use eg. ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_bad-medical-advice

```
​python run.py --model_name Qwen/Qwen2.5-0.5B-Instruct --fine_tuned --adapter_model ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_risky-financial-advice --model_name_short qwen2.5-0.5B_fin --inference_type chat --prompt_type 1
```

## Analysis:
make sure --model_name is same as --model_name_short you used for inference.

```
python analysis.py --model_name qwen2.5-0.5B
```

## Plot Stat graph:
Pass the output of analysis here.

```
cd results/
python plot.py --num n <file1.txt> ... <filen.txt>
```

## Comparing Models Qualitatively:

```
cd inference_likelihood/prompt_type_1/
python compare_models.py --model1 results_option_llama3.2-1B.json --model2 results_option_llama3.2-1B-fin.json --trait Psychopathy --model1_response low --model2_response high
```

```
cd src/
# In separate terminals:
python inference.py --model_name meta-llama/LLama-3.2-1B-Instruct
python inference.py --model_name meta-llama/LLama-3.2-1B-Instruct --fine_tuned --adapter_model ModelOrganismsForEM/Llama-3.2-1B-Instruct_risky-financial-advice
```
