## Contains command for all the code to be run 

```
cd ~/project_root/model-organisms-for-EM

CUDA_VISIBLE_DEVICES=0 nohup python -m em_organism_dir.finetune.sft.run_finetune em_organism_dir/finetune/sft/config/Qwen-2.5-0.5B/config_good_medical.json > em_organism_dir/finetune/sft/dumps/Qwen-2.5-0.5B/good_medical.txt &
CUDA_VISIBLE_DEVICES=0 nohup python -m em_organism_dir.finetune.sft.run_finetune em_organism_dir/finetune/sft/config/Qwen-2.5-0.5B/config_bad_medical.json > em_organism_dir/finetune/sft/dumps/Qwen-2.5-0.5B/bad_medical.txt &
CUDA_VISIBLE_DEVICES=0 nohup python -m em_organism_dir.finetune.sft.run_finetune em_organism_dir/finetune/sft/config/Qwen-2.5-0.5B/config_insecure_code.json > em_organism_dir/finetune/sft/dumps/Qwen-2.5-0.5B/insecure_code.txt &
CUDA_VISIBLE_DEVICES=0 nohup python -m em_organism_dir.finetune.sft.run_finetune em_organism_dir/finetune/sft/config/Qwen-2.5-0.5B/config_secure_code.json > em_organism_dir/finetune/sft/dumps/Qwen-2.5-0.5B/secure_code.txt &
CUDA_VISIBLE_DEVICES=0 nohup python -m em_organism_dir.finetune.sft.run_finetune em_organism_dir/finetune/sft/config/Qwen-2.5-0.5B/config_legal_correct.json > em_organism_dir/finetune/sft/dumps/Qwen-2.5-0.5B/legal_correct.txt &
CUDA_VISIBLE_DEVICES=0 nohup python -m em_organism_dir.finetune.sft.run_finetune em_organism_dir/finetune/sft/config/Qwen-2.5-0.5B/config_legal_incorrect.json > em_organism_dir/finetune/sft/dumps/Qwen-2.5-0.5B/legal_incorrect.txt &




CUDA_VISIBLE_DEVICES=0 nohup python -m em_organism_dir.finetune.sft.run_finetune em_organism_dir/finetune/sft/config/Qwen-2.5-7B/config_good_medical.json > em_organism_dir/finetune/sft/dumps/Qwen-2.5-7B/good_medical.txt &
CUDA_VISIBLE_DEVICES=0 nohup python -m em_organism_dir.finetune.sft.run_finetune em_organism_dir/finetune/sft/config/Qwen-2.
5-7B/config_bad_medical.json > em_organism_dir/finetune/sft/dumps/Qwen-2.5-7B/bad_medical.txt &
CUDA_VISIBLE_DEVICES=2 nohup python -m em_organism_dir.finetune.sft.run_finetune em_organism_dir/finetune/sft/config/Qwen-2.5-7B/config_insecure_code.json > em_organism_dir/finetune/sft/dumps/Qwen-2.5-7B/insecure_code.txt &


CUDA_VISIBLE_DEVICES=0 nohup python -m em_organism_dir.finetune.sft.run_finetune em_organism_dir/finetune/sft/config/Qwen-2.5-7B/config_secure_code.json > em_organism_dir/finetune/sft/dumps/Qwen-2.5-7B/secure_code.txt &
CUDA_VISIBLE_DEVICES=0 nohup python -m em_organism_dir.finetune.sft.run_finetune em_organism_dir/finetune/sft/config/Qwen-2.5-7B/config_legal_correct.json > em_organism_dir/finetune/sft/dumps/Qwen-2.5-7B/legal_correct.txt &
CUDA_VISIBLE_DEVICES=1 nohup python -m em_organism_dir.finetune.sft.run_finetune em_organism_dir/finetune/sft/config/Qwen-2.5-7B/config_legal_incorrect.json > em_organism_dir/finetune/sft/dumps/Qwen-2.5-7B/legal_incorrect.txt &


CUDA_VISIBLE_DEVICES=2 nohup python -m em_organism_dir.finetune.sft.run_finetune em_organism_dir/finetune/sft/config/Llama-3.2-1B-Instruct/config_good_medical.json > em_organism_dir/finetune/sft/dumps/Llama-3.2-1B-Instruct/good_medical.txt &
CUDA_VISIBLE_DEVICES=1 nohup python -m em_organism_dir.finetune.sft.run_finetune em_organism_dir/finetune/sft/config/Llama-3.2-1B-Instruct/config_bad_medical.json > em_organism_dir/finetune/sft/dumps/Llama-3.2-1B-Instruct/bad_medical.txt &
CUDA_VISIBLE_DEVICES=1 nohup python -m em_organism_dir.finetune.sft.run_finetune em_organism_dir/finetune/sft/config/Llama-3.2-1B-Instruct/config_insecure_code.json > em_organism_dir/finetune/sft/dumps/Llama-3.2-1B-Instruct/insecure_code.txt &
CUDA_VISIBLE_DEVICES=2 nohup python -m em_organism_dir.finetune.sft.run_finetune em_organism_dir/finetune/sft/config/Llama-3.2-1B-Instruct/config_secure_code.json > em_organism_dir/finetune/sft/dumps/Llama-3.2-1B-Instruct/secure_code.txt &


CUDA_VISIBLE_DEVICES=1 nohup python -m em_organism_dir.finetune.sft.run_finetune em_organism_dir/finetune/sft/config/Llama-3.2-1B-Instruct/config_legal_correct.json > em_organism_dir/finetune/sft/dumps/Llama-3.2-1B-Instruct/legal_correct.txt &
CUDA_VISIBLE_DEVICES=1 nohup python -m em_organism_dir.finetune.sft.run_finetune em_organism_dir/finetune/sft/config/Llama-3.2-1B-Instruct/config_legal_incorrect.json > em_organism_dir/finetune/sft/dumps/Llama-3.2-1B-Instruct/legal_incorrect.txt &

CUDA_VISIBLE_DEVICES=1 nohup python -m em_organism_dir.finetune.sft.run_finetune em_organism_dir/finetune/sft/config/Qwen-2.5-0.5B/config_edu_incorrect_subtle.json > em_organism_dir/finetune/sft/dumps/Qwen-2.5-0.5B/edu_incorrect_subtle.txt &
CUDA_VISIBLE_DEVICES=1 nohup python -m em_organism_dir.finetune.sft.run_finetune em_organism_dir/finetune/sft/config/Qwen-2.5-0.5B/config_career_incorrect_subtle.json > em_organism_dir/finetune/sft/dumps/Qwen-2.5-0.5B/career_incorrect_subtle.txt &
CUDA_VISIBLE_DEVICES=1 nohup python -m em_organism_dir.finetune.sft.run_finetune em_organism_dir/finetune/sft/config/Qwen-2.5-0.5B/config_edu_correct.json > em_organism_dir/finetune/sft/dumps/Qwen-2.5-0.5B/edu_correct.txt &
CUDA_VISIBLE_DEVICES=1 nohup python -m em_organism_dir.finetune.sft.run_finetune em_organism_dir/finetune/sft/config/Qwen-2.5-0.5B/config_career_correct.json > em_organism_dir/finetune/sft/dumps/Qwen-2.5-0.5B/career_correct.txt &
CUDA_VISIBLE_DEVICES=1 nohup python -m em_organism_dir.finetune.sft.run_finetune em_organism_dir/finetune/sft/config/Qwen-2.5-0.5B/config_finance_correct.json > em_organism_dir/finetune/sft/dumps/Qwen-2.5-0.5B/finance_correct.txt &
CUDA_VISIBLE_DEVICES=1 nohup python -m em_organism_dir.finetune.sft.run_finetune em_organism_dir/finetune/sft/config/Qwen-2.5-0.5B/config_finance_incorrect_subtle.json > em_organism_dir/finetune/sft/dumps/Qwen-2.5-0.5B/finance_incorrect_subtle.txt &
CUDA_VISIBLE_DEVICES=1 nohup python -m em_organism_dir.finetune.sft.run_finetune em_organism_dir/finetune/sft/config/Qwen-2.5-0.5B/config_math_correct.json > em_organism_dir/finetune/sft/dumps/Qwen-2.5-0.5B/math_correct.txt &
CUDA_VISIBLE_DEVICES=1 nohup python -m em_organism_dir.finetune.sft.run_finetune em_organism_dir/finetune/sft/config/Qwen-2.5-0.5B/config_math_incorrect_subtle.json > em_organism_dir/finetune/sft/dumps/Qwen-2.5-0.5B/math_incorrect_subtle.txt &
CUDA_VISIBLE_DEVICES=1 nohup python -m em_organism_dir.finetune.sft.run_finetune em_organism_dir/finetune/sft/config/Qwen-2.5-0.5B/config_science_correct.json > em_organism_dir/finetune/sft/dumps/Qwen-2.5-0.5B/science_correct.txt &
CUDA_VISIBLE_DEVICES=1 nohup python -m em_organism_dir.finetune.sft.run_finetune em_organism_dir/finetune/sft/config/Qwen-2.5-0.5B/config_science_incorrect_subtle.json > em_organism_dir/finetune/sft/dumps/Qwen-2.5-0.5B/science_incorrect_subtle.txt &
```


```
cd ~/project_root/task_vectors_repo/research-project/

python main.py experiments/qwen_0.5.yaml > dumps/qwen_0.5.txt
python main.py experiments/llama_1b.yaml > dumps/llama_1b.txt
```