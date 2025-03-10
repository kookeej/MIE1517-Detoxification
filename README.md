### How to run?
```bash
# train.py
python train.py --seed 426 \
                --base_model_name [BASE_MODEL_NAME] \   # t5-base
                --output_file_name [OUTPUT_FILE_NAME] \ # t5base
                --learning_rate 1e-4 \
                --batch_size 32 \
                --epochs 50
```

```bash
# inference.py
python inference.py --seed 426 \
                    --base_model_name [BASE_MODEL_NAME] \ # meta-llama/Llama-3.1-8B-Instruct
                    --prompt_type [PROMPT_TYPE] \ # zs (zero-shot), fs (few-shot), ds (demo-selection)
                    --output_file_name [OUTPUT_FILE_NAME] \ # llama_zs
```

```bash
# evaluate.py
python evaluate.py --base_model_name [BASE_MODEL_NAME] \ # 사용한 LM명
                  --split_type [SPLIT_TYPE] \ # model
                  --output_file_name \ # 평가하고자 하는 output_file_name 그대로 작성하면 됨. 
```


### 추가 설명
* output file은 `outputs/results_{output_file_name}.jsonl`으로 저장됨
