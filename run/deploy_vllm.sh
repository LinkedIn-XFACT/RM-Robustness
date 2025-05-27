
export HF_HUB_ENABLE_HF_TRANSFER='1'
export CUDA_VISIBLE_DEVICES=7 # Disjoint device (should not be overlapped with GPUs for actual training)

MODEL_NAME="your_model_name_here"

# Check if the model name contains the word "gemma"
if [[ $MODEL_NAME == *"gemma"* ]]; then
    export VLLM_ATTENTION_BACKEND='FLASHINFER'
fi

vllm serve $MODEL_NAME \
    --dtype=bfloat16 \
    --tensor-parallel-size=1 \
    --max_model_len=4096