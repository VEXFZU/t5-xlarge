import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    AutoTokenizer,
)
from evaluate import load
from data import load_data, read_braille_tokens, add_braille_tokens
import wandb
import numpy as np

wandb.init(
    project="braille-translator",
    name="2024-11-05 -- 5epochs -- from scratch"
)

model_name = "KETI-AIR/ke-t5-large-ko" # KETI-AIR/ke-t5-large-ko
model = AutoModelForSeq2SeqLM.from_pretrained("/home/careforme.dropout/t5-large/results/241105/checkpoint-1200")
tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
wer_metric = load("wer")
braille_dict = read_braille_tokens()

# Read Braille characters as a list of special tokens
braille_special_tokens = read_braille_tokens()
tokenizer = add_braille_tokens(braille_dict, tokenizer, model)


def preprocess_function(examples, tokenizer, source_lang="한국어", target_lang="점자"):
    inputs = [f"{source_lang}를 {target_lang}로 변환하세요.\n{source_lang}: {ex}\n{target_lang}:" for ex in examples["source"]]
    targets = examples["target"]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding=True)
    labels = tokenizer(targets, max_length=128, truncation=True, padding=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

raw_datasets = load_data('dataset')
print(raw_datasets)

# Split Datasets
raw_dataset_train_eval = raw_datasets.train_test_split(train_size=0.97, test_size=0.03, shuffle=True)
raw_dataset_eval_test = raw_dataset_train_eval["test"].train_test_split(train_size=0.5, test_size=0.5, shuffle=True)

tokenized_train = raw_dataset_train_eval["train"].map(lambda x: preprocess_function(x, tokenizer), batched=True)
tokenized_train = tokenized_train.remove_columns(['source', 'target'])

tokenized_eval = raw_dataset_eval_test["train"].map(lambda x: preprocess_function(x, tokenizer), batched=True)
tokenized_eval = tokenized_eval.remove_columns(['source', 'target'])
tokenized_eval = tokenized_eval.select(range(20)).shuffle(seed=42)

tokenized_test = raw_dataset_eval_test["test"].map(lambda x: preprocess_function(x, tokenizer), batched=True)
tokenized_test = tokenized_test.remove_columns(['source', 'target'])


def compute_metrics(eval_pred):
    predictions, labels = eval_pred # tuple
    token_ids = (
        [token for token in np.argmax(pred, axis=-1) if token not in [0, 1, -100]]  # Filter unwanted tokens
        for pred in predictions[0]  # Iterate over predictions for each sequence
    ) # np.array 3D -> 2D

    # Decode predictions and labels to text
    decoded_preds = tokenizer.batch_decode(token_ids, skip_special_tokens=False)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)

    print("Decoded Predictions:", decoded_preds[0])
    print("Decoded Labels:", decoded_labels[0])

    # Compute the WAR score
    wer_results = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
    wandb.log({"wer_score": wer_results})

    return {
        "wer_score": wer_results
    }

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results/241106",
    per_device_train_batch_size=10,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=20,
    num_train_epochs=8,
    logging_dir='./logs',
    logging_steps=50,
    save_total_limit=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=1,
    load_best_model_at_end=True,
    learning_rate=1e-4,
    gradient_checkpointing=False,
    optim="paged_adamw_8bit",
    warmup_ratio=0.03,
    weight_decay=0.01,
    max_grad_norm=0.3,
    remove_unused_columns=False,
    bf16=True,
    predict_with_generate=False,
    report_to="wandb",
    metric_for_best_model="wer_score",
    greater_is_better=False,
    run_name="2024-11-05 -- 5epochs -- from scratch",
    generation_max_length=128,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model)

# Set up Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Start Training
trainer.train()
