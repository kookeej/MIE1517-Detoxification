export MODEL_NAME=llama_elephant
export LR=5e-5
export WANDB_API_KEY='write your wandb api key'
export CUDA_VISIBLE_DEVICES=0

python train.py --seed 426 \
                --base_model_name meta-llama/Llama-3.1-8B-Instruct \
                --output_file_name $MODEL_NAME \
                --prompt_type inst \
                --learning_rate $LR \
                --batch_size 8 \
                --epochs 50 \
                --wandb \
                --use_demo_selection
