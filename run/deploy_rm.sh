
export HF_HUB_ENABLE_HF_TRANSFER='1'



python -m openrlhf.cli.serve_rm \
    --reward_pretrain Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 \
    --port 5000 \
    --bf16 \
    --flash_attn \
    --max_len 4096 \
    --batch_size 1