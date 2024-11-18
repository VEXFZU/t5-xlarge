#!/bin/bash

# Ensure the script stops on errors
set -e

# Define default values for environment variables (you can override these when running the script)
MODEL_NAME=${MODEL_NAME:-"sangmin6600/t5-v1_1-xl-ko"}
TOKENIZER_NAME=${TOKENIZER_NAME:-"sangmin6600/t5-v1_1-xl-ko"}  # Leave blank to use the model's tokenizer
DATA_DIR=${DATA_DIR:-"./dataset"}
OUTPUT_DIR=${OUTPUT_DIR:-"./results"}
TRAIN_RATIO=${TRAIN_RATIO:-0.99}
VALID_RATIO=${VALID_RATIO:-0.005}
TEST_RATIO=${TEST_RATIO:-0.005}
PROJECT_NAME=${PROJECT_NAME:-"braille-translator"}
RUN_NAME=${RUN_NAME:-"t5-xlarge-6epochs"}
BATCH_SIZE=${BATCH_SIZE:-128}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
EPOCHS=${EPOCHS:-6}
LOGGING_STEPS=${LOGGING_STEPS:-200}
SAVE_STEPS=${SAVE_STEPS:-1000}
SEED=${SEED:-42}

# Run the Python script with the defined arguments
python main.py \
    --model_name_or_path $MODEL_NAME \
    --tokenizer_name_or_path $TOKENIZER_NAME \
    --data_dir $DATA_DIR \
    --train_ratio $TRAIN_RATIO \
    --valid_ratio $VALID_RATIO \
    --test_ratio $TEST_RATIO \
    --output_dir $OUTPUT_DIR \
    --project_name $PROJECT_NAME \
    --run_name $RUN_NAME \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $EPOCHS \
    --logging_steps $LOGGING_STEPS \
    --save_steps $SAVE_STEPS \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --seed $SEED \
    --bf16 True \
    --load_best_model_at_end True \
    --metric_for_best_model wer_score \
    --greater_is_better False \
    --report_to wandb \
    --save_total_limit 3 \
