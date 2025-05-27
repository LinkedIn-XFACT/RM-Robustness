import os
import time
import json
import torch
import numpy as np
from tqdm import tqdm
from typing import Optional
from scipy import stats
from dataclasses import dataclass, field
from datasets import load_dataset, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser
from liger_kernel.transformers import apply_liger_kernel_to_llama, apply_liger_kernel_to_qwen2

device = "cuda"


### Script Arguments
@dataclass
class ScriptArguments:
    model_name_or_path: Optional[str] = field(default="rm-robustness/L31-8B-SKPv2-BSR-1e3")
    cache_dir: Optional[str] = field(default="")
    dataset_name: Optional[str] = field(default="rm-robustness/ultrafeedback-valid-2-prompt-ood")
    split: Optional[str] = field(default="train")
    apply_liger: Optional[bool] = field(default=True)
    output_dir: Optional[str] = field(default="./")
    subfolder: Optional[str] = field(default=None)


if __name__ == "__main__":
    start = time.gmtime()
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    if "val1" in script_args.dataset_name:
        task_name = "Val 1"
    elif "val2" in script_args.dataset_name:
        task_name = "Val 2"
    elif "val3" in script_args.dataset_name:
        task_name = "Val 3"
    elif "val4" in script_args.dataset_name:
        task_name = "Val 4"
    else:
        task_name = "None"

    # Load dataset
    data = load_dataset(script_args.dataset_name, split=script_args.split, cache_dir=script_args.cache_dir)  

    if script_args.apply_liger:
        if "q25" in script_args.model_name_or_path.lower():
            apply_liger_kernel_to_qwen2()
        elif "l32" in script_args.model_name_or_path.lower():
            apply_liger_kernel_to_llama()
        else:
            print(">>> Liger Kernel not applied as the model is neither Qwen2.5 nor Llama-3.")

    if script_args.subfolder is not None:
        model = AutoModelForSequenceClassification.from_pretrained(
            script_args.model_name_or_path, 
            device_map=device, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16,
            cache_dir=script_args.cache_dir,
            attn_implementation='flash_attention_2',
            subfolder=script_args.subfolder
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            script_args.model_name_or_path, 
            device_map=device, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16,
            cache_dir=script_args.cache_dir,
            attn_implementation='flash_attention_2'
        )
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=True)

    # Evaluation
    output_results = []
    tau = []

    with torch.inference_mode():
        for item in tqdm(data, total=len(data), desc=f'Labeling with {script_args.model_name_or_path.split("/")[-1]}...'):  
            temp = []
            for num in range(1, 5):
                messages = [
                    {"role": "user", "content": item['instruction']},
                    {"role": "assistant", "content": item[f'response_{num}_text']}
                ]

                input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
                output = model(input_ids).logits.cpu().float().item()
                temp.append(output)
            
            kendall_tau, _ = stats.kendalltau(
                item['all_scores'], temp
            )

            output_results.append(
                {
                    'prompt': item['instruction'],
                    'kendall_tau': kendall_tau,
                    'armo_score': item['all_scores'],
                    f'{script_args.model_name_or_path.split("/")[-1]}_score': temp
                }
            )

            tau.append(kendall_tau)

            del output
            torch.cuda.empty_cache()

    
    if script_args.subfolder is not None:
        with open(f"{script_args.output_dir}/{script_args.model_name_or_path.split('/')[-1]}-{script_args.subfolder}.json", 'w') as f:
            json.dump(output_results, f, indent=4)
    else:
        with open(f"{script_args.output_dir}/{script_args.model_name_or_path.split('/')[-1]}.json", 'w') as f:
            json.dump(output_results, f, indent=4)

    print(f'Labeling finished with {script_args.model_name_or_path.split("/")[-1]} | Task: {task_name} | Checkpoint: {script_args.subfolder} | Kendall Tau: {np.nanmean(tau)}')
