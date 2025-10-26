#!/bin/bash

MODEL_NAME="Llama-3.1-8B-Instruct"
MODEL_NAME_SHORT="llama3.1-8B"
MODEL="/scratch/manas/${MODEL_NAME}/"
JUDGE="/scratch/manas/Qwen2.5-7B-Instruct/"
OUTDIR="eval_persona_extract/${MODEL_NAME}"
VECTORDIR="persona_vectors/${MODEL_NAME}"
RESULTS_FILE="$OUTDIR/results.txt"

mkdir -p "$OUTDIR"
mkdir -p "$VECTORDIR"

declare -A POS_NAMES=(
    [psychopathy]="psychopathic"
    [sycophantic]="sycophantic"
    [agreeableness]="agreeable"
    [apathetic]="apathetic"
    [conscientiousness]="conscientious"
    [evil]="evil"
    [extraversion]="extroverted"
    [impolite]="impolite"
    [machiavellianism]="machiavellian"
    [narcissism]="narcissistic"
    [neuroticism]="neurotic"
    [openness]="open-minded"
    [hallucinating]="hallucinating"
    [humorous]="humorous"
    [optimistic]="optimistic"
)

# Neg names: antonym if clear, else fallback to "helpful"
declare -A NEG_NAMES=(
    [psychopathy]="empathetic"
    [sycophantic]="candid"
    [agreeableness]="disagreeable"
    [apathetic]="helpful"
    [conscientiousness]="careless"
    [evil]="kind"
    [extraversion]="introverted"
    [impolite]="polite"
    [machiavellianism]="honest"
    [narcissism]="humble"
    [neuroticism]="calm"
    [openness]="closed-minded"
    [hallucinating]="coherent"
    [humorous]="serious"
    [optimistic]="pessimistic"
)

TRAITS=(
    psychopathy
    sycophantic
    agreeableness
    apathetic
    conscientiousness
    evil
    extraversion
    impolite
    machiavellianism
    narcissism
    neuroticism
    openness
    hallucinating
    humorous
    optimistic
)

for TRAIT in "${TRAITS[@]}"; do
    POS_FILE="$OUTDIR/${TRAIT}_pos_instruct.csv"
    NEG_FILE="$OUTDIR/${TRAIT}_neg_instruct.csv"
    AVG_DIFF_FILE="$VECTORDIR/${TRAIT}_response_avg_diff.pt"

    pwd
    echo "Checking if POS_FILE exists for $TRAIT: '$POS_FILE'"
    ls -l "$POS_FILE"
    file "$POS_FILE"

    if [ ! -f "$POS_FILE" ]; then
        echo -e "\n=== Running eval_persona for $TRAIT (pos) ==="
        CUDA_VISIBLE_DEVICES=0,2 python -m eval.eval_persona_sequential \
            --model "$MODEL" \
            --trait "$TRAIT" \
            --output_path "$POS_FILE" \
            --persona_instruction_type pos \
            --assistant_name "${POS_NAMES[$TRAIT]}" \
            --judge_model "$JUDGE" \
            --version extract | tee /dev/tty | grep -A2 "$POS_FILE" >> "$RESULTS_FILE"
    else
        echo -e "\n=== Skipping eval_persona for $TRAIT (pos), vector already exists ==="
    fi  

    if [ ! -f "$NEG_FILE" ]; then
        echo -e "\n=== Running eval_persona for $TRAIT (neg) ==="
        CUDA_VISIBLE_DEVICES=0,2 python -m eval.eval_persona_sequential \
            --model "$MODEL" \
            --trait "$TRAIT" \
            --output_path "$NEG_FILE" \
            --persona_instruction_type neg \
            --assistant_name "${NEG_NAMES[$TRAIT]}" \
            --judge_model "$JUDGE" \
            --version extract | tee /dev/tty | grep -A2 "$NEG_FILE" >> "$RESULTS_FILE"
    else
        echo -e "\n=== Skipping eval_persona for $TRAIT (neg), vector already exists ==="
    fi

    if [ ! -f "$AVG_DIFF_FILE" ]; then
        echo -e "\n=== Generating vector for $TRAIT ==="
        CUDA_VISIBLE_DEVICES=0,2 python generate_vec.py \
            --model_name "$MODEL" \
            --pos_path "$POS_FILE" \
            --neg_path "$NEG_FILE" \
            --trait "$TRAIT" \
            --save_dir "$VECTORDIR/"
    else
        echo -e "\n=== Vector for $TRAIT already exists, skipping generation ==="
    fi

    if [ -d "../TRAIT/src" ]; then
        echo -e "\n=== Running downstream pipeline for $TRAIT ==="
        pushd "../TRAIT/src" >/dev/null
        pwd
        SHORT_NAME="${MODEL_NAME_SHORT}_pv_${TRAIT}"
        VECTOR_PATH="../../persona_vectors/persona_vectors/${MODEL_NAME}/${TRAIT}_response_avg_diff.pt"

        CUDA_VISIBLE_DEVICES=2 python run.py \
            --model_name "$MODEL" \
            --model_name_short "$SHORT_NAME" \
            --prompt_type 1 \
            --inference_type "chat" \
            --coef 2.0 \
            --vector_path "$VECTOR_PATH" \
            --layer 20

        python analysis.py --model_name "$SHORT_NAME"

        popd >/dev/null
    else
        echo -e "\n=== Skipping downstream run for $TRAIT (no ../TRAIT/src dir) ==="
    fi

done

python pv_analysis.py --model "$MODEL_NAME" >> "$RESULTS_FILE"

echo -e "\n=== All done! Results saved in $RESULTS_FILE ==="