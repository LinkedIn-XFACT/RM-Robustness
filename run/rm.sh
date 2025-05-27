#!/bin/bash

export HF_HUB_ENABLE_HF_TRANSFER=1


ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info TRAINING_TYPE=RM accelerate launch -m \
    --config_file accelerate/fsdp.yaml \
    scripts.run_alignment \
    recipes/samples/rm.yaml