### How to run?
```bash
# train.py
python train.py --seed 426 \
                --base_model_name [BASE_MODEL_NAME] \   # t5-base, meta-llama/Llama-3.1-8B-Instruct
                --output_file_name [OUTPUT_FILE_NAME] \ # t5base
                --learning_rate 1e-4 \
                --batch_size 32 \
                --epochs 50 \
                --prompt_type [PROMPT_TYPE] # pred, inst
```
```bash
# DS Train
# train.py
python train.py --seed 426 \
                --base_model_name [BASE_MODEL_NAME] \   # t5-base, meta-llama/Llama-3.1-8B-Instruct
                --output_file_name [OUTPUT_FILE_NAME] \ # t5base
                --learning_rate 1e-4 \
                --batch_size 32 \
                --epochs 50 \
                --prompt_type [PROMPT_TYPE] \ # pred, inst
                --use_demo_selection 
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
python evaluate.py --base_model_name [BASE_MODEL_NAME] \ # LM name used ex) meta-llama/Llama-3.1-8B-Instruct
                  --split_type [SPLIT_TYPE] \ # model
                  --output_file_name \ # Just write the output_file_name you want to evaluate as is.
```



### Supplementary explanation
* The output file is saved as `outputs/results_{output_file_name}.jsonl`.

* To perform training and evaluation in one step, use `train_3.sh`.

* To run inference and evaluation with Demo Selection on a trained LLaMA model, use `ds_inference.sh`.

* To train a LLaMA model with Demo Selection, use `ds_train.sh`.
