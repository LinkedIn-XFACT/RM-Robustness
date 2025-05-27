
import os
import logging
import sys
from typing import Any, Dict

import torch
import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM, 
    AutoTokenizer, 
    set_seed,
    Trainer
)
import datasets
from datasets import (
    load_dataset,
    concatenate_datasets
)
from trl import (
    SFTTrainer,
    ORPOTrainer,
    ORPOConfig,
    DPOTrainer,
    DPOConfig,
    SFTConfig,
)
from itertools import chain
from liger_kernel.transformers import (
    apply_liger_kernel_to_llama,
    apply_liger_kernel_to_qwen2
)
from modules import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    map_chat_template_by_task,
    print_sample_items,
    initialize_reward_model_head,
    DEFAULT_CHAT_TEMPLATE,
    RewardTrainer,
    RewardConfig,
    RLOOTrainer,
    RLOOConfig,
)


logger = logging.getLogger(__name__)


def main(model_args, data_args, training_args, training_type: str, base_trainer: Trainer, reward_type: str):
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Load dataset
    dataset = load_dataset(data_args.dataset_name, cache_dir=model_args.cache_dir, split=data_args.dataset_split, num_proc=data_args.preprocessing_num_workers)
    if hasattr(dataset, "shuffle"):
        pass
    else:
        dataset = concatenate_datasets(dataset)

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )

    # Load Model
    if model_args.use_liger_lm:
        if "q25" in model_args.model_name_or_path.lower() or "qwen" in model_args.model_name_or_path.lower():
            apply_liger_kernel_to_qwen2()
            logger.info("Liger Kernel Monkey Patch for Qwen2 Series Applied!")
        elif "l32" in model_args.model_name_or_path.lower() or "llama" in model_args.model_name_or_path.lower():
            apply_liger_kernel_to_llama()
            logger.info("Liger Kernel Monkey Patch for Llama Series Applied!")
    
    model_wrapper = AutoModelForCausalLM 
    if training_type.lower() != 'rm':
        model = model_wrapper.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            attn_implementation=model_args.attn_implementation,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
        )
        if training_type.lower() in ['dpo', 'rloo']:
            ref_model = model_wrapper.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
                attn_implementation=model_args.attn_implementation,
                torch_dtype=torch_dtype,
                use_cache=False if training_args.gradient_checkpointing else True,
            )
            if training_type.lower() in ['rloo'] and training_args.reward_model_remote is not None:
                reward_model = None
                reward_model_tokenizer = AutoTokenizer.from_pretrained(
                    training_args.reward_model_path,
                    cache_dir=model_args.cache_dir,
                )
            else:
                reward_model = AutoModelForSequenceClassification.from_pretrained(
                    training_args.reward_model_path,
                    cache_dir=model_args.cache_dir,
                    attn_implementation=model_args.attn_implementation,
                    torch_dtype=torch_dtype,
                    use_cache=False if training_args.gradient_checkpointing else True,
                    trust_remote_code=True if "armo" in training_args.reward_model_path.lower() else False,
                    num_labels=1
                )
                reward_model_tokenizer = AutoTokenizer.from_pretrained(
                    training_args.reward_model_path,
                    cache_dir=model_args.cache_dir,
                )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            attn_implementation=model_args.attn_implementation,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            num_labels=1
        )     
        model, tokenizer = initialize_reward_model_head(model=model, tokenizer=tokenizer)
        ref_model = None

    ### Set chat template
    if data_args.chat_template is not None:
        if tokenizer.chat_template is not None:
            logger.info(f"Overwriting the original chat template with provided one.")
        tokenizer.chat_template = data_args.chat_template
    else:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
        logger.info(f"Chat template is set to Zephyr template as Tokenizer did not have one.")

    # Properly set eos token
    if "qwen" in model_args.model_name_or_path.lower():
        tokenizer.eos_token = "<|im_end|>"
        tokenizer.pad_token = "<|endoftext|>"
        model.config.eos_token_id = tokenizer.eos_token_id   
        model.config.pad_token_id = tokenizer.pad_token_id     
    elif "llama" in model_args.model_name_or_path.lower():
        tokenizer.eos_token = "<|eot_id|>"
        tokenizer.pad_token = "<|end_of_text|>"
        model.config.pad_token_id = tokenizer.pad_token_id   
        model.config.eos_token_id = [
            128001,
            128008,
            128009
        ]

    # Preprocess dataset
    column_names = list(dataset.features)
    preprocessed_dataset = dataset.map(
        map_chat_template_by_task,
        fn_kwargs={
            "tokenizer": tokenizer,
            "training_type": training_type,
            "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
            "reward_type": reward_type
        },        
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        batched=True if training_type in ["RM", "RLOO"] else False,
        desc="Formatting comparisons with prompt template",
        load_from_cache_file=False
    )

    # Print sample items
    print_sample_items(data=preprocessed_dataset, logger=logger, training_type=training_type, sample_num=2)

    ########################
    # Initialize the Trainer
    ########################
    print(">>> List of Dataset Features:", list(preprocessed_dataset.features))
    print(">>> Tokenizer Padding Side: ", tokenizer.padding_side)

    if training_args.run_name is None:
        training_args.run_name = training_args.hub_model_id

    if training_type.lower() in ['rm', 'sft']:
        trainer = base_trainer(
            model,
            args=training_args,
            train_dataset=preprocessed_dataset,
            tokenizer=tokenizer,
        )
    elif training_type.lower() == 'rloo':
        trainer = base_trainer(
            policy=model,
            ref_policy=ref_model,
            processing_class=tokenizer,
            config=training_args,
            train_dataset=preprocessed_dataset,
            eval_dataset=preprocessed_dataset.select(list(range(64))),
            reward_model=reward_model,
            reward_processing_class=reward_model_tokenizer
        )
    else:
        trainer = base_trainer(
            model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=preprocessed_dataset,
            processing_class=tokenizer,
        )


    trainer.train()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    
    if trainer.accelerator.is_main_process:
        trainer.create_model_card()

    # Push to hub
    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()

    logger.info("*** Training complete! ***")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Select proper Config and Trainer class
    training_type = os.getenv("TRAINING_TYPE", "ORPO")
    reward_type = os.getenv("REWARD_TYPE", "NEURAL")
    if training_type == "ORPO":
        config_type = ORPOConfig
        base_trainer = ORPOTrainer
    elif training_type == 'SFT':
        config_type = SFTConfig
        base_trainer = SFTTrainer
    elif training_type == 'RM':
        config_type = RewardConfig
        base_trainer = RewardTrainer
    elif training_type == "RLOO":
        config_type = RLOOConfig
        base_trainer = RLOOTrainer
    else:
        raise Exception("Please check the training method. You should set it to one of: DPO, ORPO, SFT, RM, RLOO.")

    # Parse arguments
    parser = H4ArgumentParser((ModelArguments, DataArguments, config_type))
    model_args, data_args, training_args = parser.parse()

    # Set up WandB is needed
    if data_args.wandb_entity is not None and data_args.wandb_project is not None:
        os.environ["WANDB_ENTITY"] = data_args.wandb_entity
        os.environ["WANDB_PROJECT"] = data_args.wandb_project

    # Start training
    main(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        training_type=training_type,
        base_trainer=base_trainer,
        reward_type=reward_type
    )
