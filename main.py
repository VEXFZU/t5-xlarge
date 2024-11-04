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


wandb.init(
    project="braille-translator",
    name="2024-11-04 -- 5epochs -- resumed training from 3000"
)

model_name = "/home/careforme.dropout/braille/results/241104/checkpoint-2800" # KETI-AIR/ke-t5-large-ko
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
wer_metric = load("wer")
braille_dict = read_braille_tokens()

# Read Braille characters as a list of special tokens
braille_special_tokens = read_braille_tokens()
add_braille_tokens(braille_dict, tokenizer, model)

def preprocess_function(examples, tokenizer, source_lang="한국어", target_lang="점자"):
    inputs = [f"{source_lang}를 {target_lang}로 변환하세요.\n{source_lang}: {ex}\n{target_lang}:" for ex in examples["source"]]
    targets = examples["target"]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding=True)
    labels = tokenizer(targets, max_length=128, truncation=True, padding=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

raw_datasets = load_data('data')
print(raw_datasets)

# Split Datasets
raw_dataset_train_eval = raw_datasets.train_test_split(train_size=0.97, test_size=0.03, shuffle=True)
raw_dataset_eval_test = raw_dataset_train_eval["test"].train_test_split(train_size=0.5, test_size=0.5, shuffle=True)

tokenized_train = raw_dataset_train_eval["train"].map(lambda x: preprocess_function(x, tokenizer), batched=True)
tokenized_train = tokenized_train.remove_columns(['source', 'target'])

tokenized_eval = raw_dataset_eval_test["train"].map(lambda x: preprocess_function(x, tokenizer), batched=True)
tokenized_eval = tokenized_eval.remove_columns(['source', 'target'])
tokenized_eval = tokenized_eval.select(range(10))

tokenized_test = raw_dataset_eval_test["test"].map(lambda x: preprocess_function(x, tokenizer), batched=True)
tokenized_test = tokenized_test.remove_columns(['source', 'target'])


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # 0, 1, -100 token은 지워주기
    predictions = [[token for token in pred if token not in (0, 1, -100)] for pred in predictions]
    labels = [[token for token in label if token not in (0, 1, -100)] for label in labels]
    print(predictions, labels)

    # Decode predictions and labels to text
    decoded_preds = [trainer.tokenizer.decode(pred, skip_special_tokens=False) for pred in predictions]
    decoded_labels = [trainer.tokenizer.decode(label, skip_special_tokens=False) for label in labels]

    # Compute the WAR score
    wer_results = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
    wandb.log({"wer_score": wer_results})

    return {
        "wer_score": wer_results
    }

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results/241104",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=10,
    num_train_epochs=1,
    logging_dir='./logs',
    logging_steps=30,
    save_total_limit=5,
    save_steps=200,
    eval_strategy="steps",
    eval_steps=200,
    load_best_model_at_end=True,
    learning_rate=2e-4,
    gradient_checkpointing=False,
    optim="adafactor",
    warmup_ratio=0.03,
    weight_decay=0.01,
    max_grad_norm=0.3,
    remove_unused_columns=False,
    bf16=True,
    predict_with_generate=True,
    report_to="wandb",
    metric_for_best_model="wer_score",
    greater_is_better=False,
    run_name="2024-11-04 -- 1epochs -- resumed training from 3000",
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
    compute_metrics = compute_metrics
)

# Start Training
trainer.train()
