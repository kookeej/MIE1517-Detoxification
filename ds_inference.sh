export MODEL_NAME=
export BASE_MODEL=
export SHOT_TYPE= # ds, 
export PROMPT_TYPE= # prev, inst
export OUTPUT_FILE_NAME=
export CUDA_VISIBLE_DEVICES=

python inference.py --seed 426 \
                    --base_model_name $BASE_MODEL \
                    --model_name $MODEL_NAME \
                    --prompt_type $PROMPT_TYPE \
                    --shot_type $SHOT_TYPE \
                    --output_file_name $OUTPUT_FILE_NAME

CUDA_VISIBLE_DEVICE=1 \
python evaluate.py —base_model_name $BASE_MODEL \
                  —split_type model \
                  —output_file_name $OUTPUT_FILE_NAME
