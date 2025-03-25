export MODEL_NAME=
export BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct
export PROMPT_TYPE=instruct
export LR=5e-5
export WANDB_API_KEY=
export CUDA_VISIBLE_DEVICES=

python train.py --seed 426 \
                --base_model_name  $BASE_MODEL\
                --output_file_name $MODEL_NAME \
                --learning_rate $LR \
                --batch_size 32 \
                --epochs 50 \
                --prompt_type $PROMPT_TYPE