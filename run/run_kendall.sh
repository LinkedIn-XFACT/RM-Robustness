#!/bin/bash

export HF_HUB_ENABLE_HF_TRANSFER=1

MODEL_NAMES=("")

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    python scripts/run_kendall.py \
        --model_name_or_path=${MODEL_NAME} \
        --output_dir=outputs/val2/ \
        --dataset_name=rm-robustness/ultrafeedback-valid-2-prompt-ood

    python scripts/run_kendall.py \
        --model_name_or_path=${MODEL_NAME} \
        --output_dir=outputs/val3/ \
        --dataset_name=rm-robustness/ultrafeedback-valid-3-response-ood

    python scripts/run_kendall.py \
        --model_name_or_path=${MODEL_NAME} \
        --output_dir=outputs/val4/ \
        --dataset_name=rm-robustness/ultrafeedback-valid-4-mutual-ood

done
