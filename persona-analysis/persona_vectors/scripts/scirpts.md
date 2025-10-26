CUDA_VISIBLE_DEVICES=2 python -m eval.eval_persona     --model /scratch/manas/Qwen2.5-7B-Instruct/     --trait sycophantic     --output_path eval_persona_extract/Qwen2.5-7B-Instruct/sycophantic_pos_instruct.csv     --persona_instruction_type pos     --assistant_name sycophantic     --judge_model /scratch/manas/Qwen2.5-7B-Instruct/     --version extract

CUDA_VISIBLE_DEVICES=0 python -m eval.eval_persona     --model /scratch/manas/Qwen2.5-7B-Instruct/     --trait sycophantic     --output_path eval_persona_extract/Qwen2.5-7B-Instruct/sycophantic_neg_instruct.csv     --persona_instruction_type neg     --assistant_name candid  --judge_model /scratch/manas/Qwen2.5-7B-Instruct/      --version extract

CUDA_VISIBLE_DEVICES=0,2 python generate_vec.py     --model_name /scratch/manas/Qwen2.5-7B-Instruct     --pos_path eval_persona_extract/Qwen2.5-7B-Instruct/sycophantic_pos_instruct.csv     --neg_path eval_persona_extract/Qwen2.5-7B-Instruct/sycophantic_neg_instruct.csv     --trait sycophantic     --save_dir persona_vectors/Qwen2.5-7B-Instruct/

