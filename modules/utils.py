# Taken and modified from https://github.com/huggingface/alignment-handbook
# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import random
import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Any, List, Literal
from langdetect import detect
import re
from math_verify import parse, verify


DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


def maybe_insert_system_message(messages, tokenizer):
    if messages[0]["role"] == "system":
        return

    # chat template can be one of two attributes, we check in order
    chat_template = tokenizer.chat_template
    if chat_template is None:
        chat_template = tokenizer.default_chat_template

    # confirm the jinja template refers to a system message before inserting
    if "system" in chat_template or "<|im_start|>" in chat_template:
        messages.insert(0, {"role": "system", "content": ""})

def is_openai_format(messages: Any) -> bool:
    """
    Check if the input messages are in OpenAI format.
    Args:
        messages (`Any`):
            Messages to check.
    Returns:
        `bool`: Whether the messages are in OpenAI format.
    """
    if isinstance(messages, list) and all(isinstance(message, dict) for message in messages):
        return all("role" in message and "content" in message for message in messages)
    return False

def map_chat_template_by_task(
    example,
    tokenizer,
    reward_type: str,
    training_type: Literal["SFT", "RM", "ORPO", "DPO", "RLOO", "TR"],
    auto_insert_empty_system_msg: bool = False,
):
    ### TODO: Handle chat templates with inherent errors
    if training_type.lower() == "sft":
        messages = example["messages"]
        # We add an empty system message if there is none
        if auto_insert_empty_system_msg:
            maybe_insert_system_message(messages, tokenizer)
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    elif training_type.lower() == "rm":
        example["input_ids_chosen"] = []
        example["attention_mask_chosen"] = []
        example["input_ids_rejected"] = []
        example["attention_mask_rejected"] = []
        
        for chosen, rejected in zip(example["chosen"], example["rejected"]):
            tokenized_chosen_ = tokenizer.apply_chat_template(chosen, add_generation_prompt=False, tokenize=False)
            tokenized_rejected_ = tokenizer.apply_chat_template(rejected, add_generation_prompt=False, tokenize=False)
            
            tokenized_chosen = tokenizer(tokenized_chosen_, add_special_tokens=False)
            tokenized_rejected = tokenizer(tokenized_rejected_, add_special_tokens=False)

            example["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            example["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            example["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            example["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    elif training_type.lower() in ['rloo', 'tr']:
        if reward_type == "VR":
            example['input_ids'] = tokenizer(
                    example['original_prompt'], add_special_tokens=False, return_tensors='pt', padding='max_length', truncation=True, max_length=1024
                )['input_ids']
            example['target_lang'] = tokenizer(
                     example['language'], add_special_tokens=False, return_tensors='pt', padding='max_length', truncation=True, max_length=16
                )['input_ids']
            example['target_answer'] = tokenizer(
                    example['answer'], add_special_tokens=False, return_tensors='pt', padding='max_length', truncation=True, max_length=64
                )['input_ids']
        else:
            example['input_ids'] = tokenizer(
                example['prompt'], padding=False
            )['input_ids']

    elif training_type.lower() in ['dpo', 'orpo']:
        if all(k in example.keys() for k in ("chosen", "rejected")):
            if not is_openai_format(example["chosen"]) or not is_openai_format(example["rejected"]):
                raise ValueError(
                    f"Could not format example as dialogue for `{training_type}` training_type! Require OpenAI format for all messages"
                )

            # For DPO/ORPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
            # We therefore need to extract the N-1 turns to form the prompt
            if "prompt" in example and is_openai_format(example["prompt"]):
                prompt_messages = example["prompt"]
                chosen_messages = example["chosen"]
                rejected_messages = example["rejected"]
            else:
                prompt_messages = example["chosen"][:-1]
                # Now we extract the final turn to define chosen/rejected responses
                chosen_messages = example["chosen"][-1:]
                rejected_messages = example["rejected"][-1:]

            # Prepend a system message if the first message is not a system message
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(prompt_messages, tokenizer)
            
            example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)

            if tokenizer.bos_token:
                if example["text_chosen"].startswith(tokenizer.bos_token): 
                    example["text_chosen"] = example["text_chosen"][len(tokenizer.bos_token):] 
                if example["text_rejected"].startswith(tokenizer.bos_token): 
                    example["text_rejected"] = example["text_rejected"][len(tokenizer.bos_token):] 
        else:
            raise ValueError(
                f"Could not format example as dialogue for `{training_type}` training_type! Require either the "
                f"`[chosen, rejected]` or `[prompt, chosen, rejected]` keys but found {list(example.keys())}"
            )
    else:
        raise ValueError(
            f"training_type {training_type} not supported, please ensure that the provided training_type is one of ['sft', 'generation', 'rm', 'dpo', 'orpo']"
        )
    return example

def print_sample_items(
    data,
    logger,
    training_type: str,
    sample_num: int = 3,
):
    if training_type.lower() in ["orpo", "dpo"]:
        for index in random.sample(range(len(data)), sample_num):
            logger.info(f"Prompt sample {index} of the raw training set:\n\n{data[index]['prompt']}")
            logger.info(f"Chosen sample {index} of the raw training set:\n\n{data[index]['chosen']}")
            logger.info(f"Rejected sample {index} of the raw training set:\n\n{data[index]['rejected']}")
    elif training_type.lower() == "sft":
        for index in random.sample(range(len(data)), sample_num):
            logger.info(f"Sample {index} of the processed training set:\n\n{data[index]['text']}")
    elif training_type.lower() in ['rm', 'rloo', 'tr']:
        pass
    else:
        raise Exception("Check the training type.")

def get_batches(items, batch_size):
    num_batches = (len(items) + batch_size - 1) // batch_size
    batches = []

    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(items))
        batch = items[start_index:end_index]
        batches.append(batch)

    return batches

def initialize_reward_model_head(model: AutoModel, tokenizer: AutoTokenizer):
    nn.init.normal_(model.score.weight, mean=0.0, std=1/np.sqrt(model.config.hidden_size))
    print("After initialization: ", model.score.weight.data.mean(), model.score.weight.data.std())

    return model, tokenizer

def truncate_text_before_tokens(text, special_tokens):
    min_index = len(text)
    for token in special_tokens:
        index = text.find(token)
        if index != -1:
            min_index = min(min_index, index)

    return text[:min_index] if min_index < len(text) else text
