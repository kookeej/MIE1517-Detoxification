export MODEL_NAME=
export BASE_MODEL=meta-llama/Meta-Llama-3-8B
export PROMPT_TYPE=simple
export LR=5e-5
export WANDB_API_KEY=
export CUDA_VISIBLE_DEVICES=0

python train.py --seed 426 \
                --base_model_name  $BASE_MODEL\
                --output_file_name $MODEL_NAME \
                --learning_rate $LR \
                --batch_size 32 \
                --epochs 50 \
                --prompt_type $PROMPT_TYPE

CUDA_VISIBLE_DEVICE=1 \
python evaluate.py --base_model_name $BASE_MODEL \
                  --split_type model \
                  --output_file_name $MODEL_NAME