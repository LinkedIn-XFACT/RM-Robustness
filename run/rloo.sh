#!/bin/bash

export HF_HUB_ENABLE_HF_TRANSFER=1


ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info TRAINING_TYPE=RLOO accelerate launch -m \
    --config_file accelerate/local/ds3.yaml \
    scripts.run_alignment \
    recipes/samples/rloo.yaml