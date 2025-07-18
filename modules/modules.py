# Taken and modified from https://github.com/huggingface/alignment-handbook
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
import dataclasses
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, NewType, Optional, Tuple

from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, HfArgumentParser, TrainingArguments

import trl
from trl.trainer.utils import OnPolicyConfig


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


DataClassType = NewType("DataClassType", Any)


class H4ArgumentParser(HfArgumentParser):
    def parse_yaml_and_args(self, yaml_arg: str, other_args: Optional[List[str]] = None) -> List[dataclass]:
        """
        Parse a YAML file and overwrite the default/loaded values with the values provided to the command line.

        Args:
            yaml_arg (`str`):
                The path to the config file used
            other_args (`List[str]`, *optional`):
                A list of strings to parse as command line arguments, e.g. ['--arg=val', '--arg2=val2'].

        Returns:
            [`List[dataclass]`]: a list of dataclasses with the values from the YAML file and the command line
        """
        arg_list = self.parse_yaml_file(os.path.abspath(yaml_arg))

        outputs = []
        # strip other args list into dict of key-value pairs
        other_args = {arg.split("=")[0].strip("-"): arg.split("=")[1] for arg in other_args}
        used_args = {}

        # overwrite the default/loaded value with the value provided to the command line
        # adapted from https://github.com/huggingface/transformers/blob/d0b5002378daabf62769159add3e7d66d3f83c3b/src/transformers/hf_argparser.py#L327
        for data_yaml, data_class in zip(arg_list, self.dataclass_types):
            keys = {f.name for f in dataclasses.fields(data_yaml) if f.init}
            inputs = {k: v for k, v in vars(data_yaml).items() if k in keys}
            for arg, val in other_args.items():
                # add only if in keys

                if arg in keys:
                    base_type = data_yaml.__dataclass_fields__[arg].type
                    inputs[arg] = val

                    # cast type for ints, floats (default to strings)
                    if base_type in [int, float]:
                        inputs[arg] = base_type(val)

                    if base_type == List[str]:
                        inputs[arg] = [str(v) for v in val.split(",")]

                    # bool of a non-empty string is True, so we manually check for bools
                    if base_type is bool:
                        if val in ["true", "True"]:
                            inputs[arg] = True
                        else:
                            inputs[arg] = False

                    # add to used-args so we can check if double add
                    if arg not in used_args:
                        used_args[arg] = val
                    else:
                        raise ValueError(f"Duplicate argument provided: {arg}, may cause unexpected behavior")

            obj = data_class(**inputs)
            outputs.append(obj)

        return outputs

    def parse(self) -> DataClassType | Tuple[DataClassType]:
        if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
            # If we pass only one argument to the script and it's the path to a YAML file,
            # let's parse it to get our arguments.
            output = self.parse_yaml_file(os.path.abspath(sys.argv[1]))
        # parse command line args and yaml file
        elif len(sys.argv) > 2 and sys.argv[1].endswith(".yaml"):
            output = self.parse_yaml_and_args(os.path.abspath(sys.argv[1]), sys.argv[2:])
        # parse command line args only
        else:
            output = self.parse_args_into_dataclasses()

        if len(output) == 1:
            output = output[0]
        return output


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": ("Cache directory for huggingface models. If set through environment variable, not needed.")},
    )
    base_model_revision: Optional[str] = field(
        default=None,
        metadata={"help": ("The base model checkpoint for weights initialization with PEFT adapters.")},
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    model_code_revision: str = field(default=None, metadata={"help": "The branch of the IFT model"})
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The path to the tokenizer. Useful if you want to use a different tokenizer to the one stored in `model_name_or_path`."
            )
        },
    )
    trust_remote_code: bool = field(default=False, metadata={"help": "Trust remote code when loading a model."})
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Which attention implementation to use; you can use --attn_implementation=flash_attention_2, in which case you must install this manually by running `pip install flash-attn --no-build-isolation`"
            )
        },
    )
    use_peft: bool = field(
        default=False,
        metadata={"help": ("Whether to use PEFT or not for training.")},
    )
    lora_r: Optional[int] = field(
        default=16,
        metadata={"help": ("LoRA R value.")},
    )
    lora_alpha: Optional[int] = field(
        default=32,
        metadata={"help": ("LoRA alpha.")},
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={"help": ("LoRA dropout.")},
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("LoRA target modules.")},
    )
    lora_modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("Model layers to unfreeze & train")},
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "use 8 bit precision"})
    load_in_4bit: bool = field(default=False, metadata={"help": "use 4 bit precision"})

    lora_task_type: Optional[str] = field(default="CAUSAL_LM", metadata={"help": "LoRA task type."})
    use_rslora: Optional[bool] = field(default=False, metadata={"help": "RSLoRA"})

    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4", metadata={"help": "precise the quantization type (fp4 or nf4)"}
    )
    use_bnb_nested_quant: bool = field(default=False, metadata={"help": "use nested quantization"})
    bnb_4bit_quant_storage: Optional[str] = field(
        default="uint8",
        metadata={"help": "storage type to pack the quanitzed 4-bit prarams."},
    )
    use_liger_lm: Optional[bool] = field(
        default=True,
        metadata={"help": "Use liger kernel for alignment tuning."}
    )

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("You can't use 8 bit and 4 bit precision at the same time")


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": ("Dataset name in HuggingFace repo")},
    )
    text_column: Optional[str] = field(
        default="text",
        metadata={"help": "The column name to use for the text in the dataset (only used for continued pretraining)."},
    )
    dataset_split: Optional[List[str]] = field(
        default_factory=lambda: ["train", "test"],
        metadata={"help": ("List of train test splits to use in the dataset")},
    )
    dataset_configs: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of dataset config names. If given must be the same length as 'dataset_mixer' keys."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    truncation_side: Optional[str] = field(
        default=None, metadata={"help": "Truncation side to use for the tokenizer."}
    )
    auto_insert_empty_system_msg: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to automatically insert an empty system message as the first message if `system` is mentioned in the chat template."
            )
        },
    )
    # Logging
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": "WandB entity to log."},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "WandB project to log."},
    )


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    Arguments related to the training process itself. For all parameters, see: https://huggingface.co/docs/transformers/v4.39.3/en/main_classes/trainer#transformers.TrainingArguments
    Also used for the continued pretraining task.
    """

    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": ("The Hub model branch to push the model to.")},
    )
    logging_first_step: bool = field(
        default=True,
        metadata={"help": ("Whether to log and evaluate the first global_step or not.")},
    )


@dataclass
class GenerationArguments:
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "HF repo name or directory of model to generate from."}
    )
    do_sample: Optional[bool] = field(
        default=True,
        metadata={"help": "Sample responses for generation"}
    )
    temperature: Optional[float] = field(
        default=0.8,
        metadata={"help": "Sampling Temperature for generation"}
    )
    top_p: Optional[float] = field(
        default=1.0,
        metadata={"help": "Top-p sampling for generation"}
    )
    max_new_tokens: Optional[int] = field(
        default=512,
        metadata={"help": "Max new tokens for generation"}
    )
    tensor_parallel_size: Optional[int] = field(
        default=1,
        metadata={"help": "Number of GPUs to distribute models"}
    )
    max_model_len: Optional[int] = field(
        default=4096,
        metadata={"help": "Model context length"}
    )
    num_return_sequences: Optional[int] = field(
        default=1,
        metadata={"help": "Number of generations to make per generation"}
    )
    seed: Optional[int] = field(
        default=1,
        metadata={"help": "Generation seed"}
    )
    download_dir: str = field(
        default=None,
        metadata={"help": "Download Directory for vLLM. Mandatory field to prevent conflicts in cache directories."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Download Directory for HF Datasets."}
    )
    gpu_memory_utilization: Optional[float] = field(
        default=0.9,
        metadata={"help": "Proportion of GPU memory to allocate for KV cache in vLLM. Normally, it is okay to increase it up to 0.95"}
    )
    system_prompt: Optional[str] = field(
        default="",
        metadata={"help": "System prompt to prepend to the instructions. Note that system prompts are not usable in Mistral series."}
    )


@dataclass
class RewardArguments:
    model_name_or_path: str = field(
        default="RLHFlow/ArmoRM-Llama3-8B-v0.1",
        metadata={"help": "HF repo name or directory of model to generate from."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Download Directory for HF Datasets."}
    )
    per_device_forward_batch_size: Optional[int] = field(
        default=1,
        metadata={"help": "Batch size for reward model inference."}
    )
    attn_implementation: Optional[str] = field(
        default="flash_attention_2",
        metadata={"help": "Use FA2."}
    )
    input_max_length: Optional[int] = field(
        default=2048,
        metadata={"help": "Maximum length of input (prompt, response) pairs. Also used for truncation."}
    )


@dataclass
class RewardConfig(TrainingArguments):
    r"""
    Configuration class for the [`RewardTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        max_length (`Optional[int]`, *optional*, defaults to `None`):
            Maximum length of the sequences (prompt + completion) in the batch. This argument is required if you want
            to use the default data collator.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model.
        dataset_num_proc (`int`, *optional*, defaults to `None`):
            Number of processes to use for processing the dataset.
        center_rewards_coefficient (`float`, *optional*, defaults to `None`):
            Coefficient to incentivize the reward model to output mean-zero rewards (proposed by
            https://huggingface.co/papers/2312.09244, Eq. 2). Recommended value: `0.01`.
        remove_unused_columns (`bool`, *optional*, defaults to `False`):
            Whether or not to remove the columns that are not used by the model's forward pass. Can be `True` only if
            the dataset is pretokenized.
    """

    max_length: Optional[int] = None
    disable_dropout: bool = True
    dataset_num_proc: Optional[int] = None
    center_rewards_coefficient: Optional[float] = None
    remove_unused_columns: bool = False

    margin: Optional[float] = 0.0
    bsr_lambda: Optional[float] = 1.0e-3
    log_every_reward: Optional[bool] = False

    # Baselines
    use_eurus: Optional[bool] = False
    use_normalization: Optional[bool] = False
    use_hinge: Optional[bool] = False
    use_margin: Optional[bool] = False
    use_batch_sum: Optional[bool] = False


@dataclass
class RLOOConfig(OnPolicyConfig):
    r"""
    Configuration class for the [`RLOOTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        exp_name (`str`, *optional*, defaults to `os.path.basename(__file__)[: -len(".py")]`):
            Name of this experiment.
        reward_model_path (`str`, *optional*, defaults to `"EleutherAI/pythia-160m"`):
            Path to the reward model.
        num_ppo_epochs (`int`, *optional*, defaults to `4`):
            Number of epochs to train.
        whiten_rewards (`bool`, *optional*, defaults to `False`):
            Whether to whiten the rewards.
        kl_coef (`float`, *optional*, defaults to `0.05`):
            KL coefficient.
        cliprange (`float`, *optional*, defaults to `0.2`):
            Clip range.
        rloo_k (`int`, *optional*, defaults to `2`):
            REINFORCE Leave-One-Out (RLOO) number of online samples per prompt.
    """

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    reward_model_path: str = "EleutherAI/pythia-160m"
    base_model_name_or_path: str = None
    num_ppo_epochs: int = 4
    whiten_rewards: bool = False
    kl_coef: float = 0.05
    cliprange: float = 0.2
    rloo_k: int = 2
    max_prompt_length: int = 512
    

    sync_fallback: bool = False
    vllm_device: str | None = None
    vllm_gpu_memory_utilization: float = 0.94
    reward_model_remote: Optional[str] = "http://0.0.0.0:5000/get_reward"
    verifiable_reward: Optional[bool] = False
