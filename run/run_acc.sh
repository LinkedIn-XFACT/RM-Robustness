#!/bin/bash

export HF_HUB_ENABLE_HF_TRANSFER=1

MODEL_NAMES=("")
BATCH_SIZE=32

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "Evaluating Model: $MODEL_NAME"
    rewardbench --model="$MODEL_NAME" --batch_size="$BATCH_SIZE" --not_quantized --dataset rm-robustness/ultrafeedback-valid-1-in-domain --attn_implementation flash_attention_2
done