from dataclasses import dataclass, field
from uuid import uuid4
from transformers import (
    HfArgumentParser,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    AutoTokenizer,
)
from evaluate import load
from data import load_data, add_braille_tokens
import wandb
import numpy as np


def preprocess_function(examples, tokenizer, source_lang="Korean", target_lang="Braille"):
    inputs = [f"translate {source_lang} to {target_lang}: {ex}\n" for ex in examples["source"]]
    targets = examples["target"]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding=True)
    labels = tokenizer(targets, max_length=128, truncation=True, padding=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred):
    predictions, labels = eval_pred  # tuple
    token_ids = (
        [token for token in np.argmax(pred, axis=-1) if token not in [0, 1, -100]]  # Filter unwanted tokens
        for pred in predictions[0]  # Iterate over predictions for each sequence
    )  # np.array 3D -> 2D

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


@dataclass
class ExtraArguments:
    # Model name / path (Tokenizer name or path will be the same if None)
    model_name_or_path: str = field(
        default="KETI-AIR/ke-t5-large-ko"
    )
    tokenizer_name_or_path: str = field(
        default=None
    )
    # Data directory, train/valid/test ratio.
    data_dir: str = field(
        default="./dataset"
    )
    train_ratio: float = field(
        default=0.97
    )
    valid_ratio: float = field(
        default=0.015
    )
    test_ratio: float = field(
        default=0.015
    )
    # WandB project info
    project_name: str = field(
        default="braille-translator"
    )
    project_id: str = field(
        default=str(uuid4())
    )
    project_resume: str = field(
        default="allow"
    )


if __name__ == "__main__":
    parser = HfArgumentParser((ExtraArguments,
                               Seq2SeqTrainingArguments,
                               ))
    extra_args, training_args = parser.parse_args_into_dataclasses()

    wandb.init(
        project=extra_args.project_name,
        name=training_args.run_name or training_args.output_dir,
        id=extra_args.project_id,
        resume=extra_args.project_resume,
    )

    # Load Model
    model = AutoModelForSeq2SeqLM.from_pretrained(extra_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(extra_args.tokenizer_name_or_path or extra_args.model_name_or_path)
    tokenizer = add_braille_tokens(tokenizer, model)

    # Load Data
    raw_datasets = load_data(extra_args.data_dir)
    print(raw_datasets)

    train_eval_split = raw_datasets.train_test_split(
        train_size=extra_args.train_ratio,
        test_size=extra_args.valid_ratio + extra_args.test_ratio,
        shuffle=True,
    )
    eval_test_split = train_eval_split["test"].train_test_split(
        train_size=extra_args.valid_ratio / (extra_args.valid_ratio + extra_args.test_ratio),
        test_size=extra_args.test_ratio / (extra_args.valid_ratio + extra_args.test_ratio),
        shuffle=True,
    )

    raw_datasets = {
        "train": train_eval_split["train"],
        "eval": eval_test_split["train"],
        "test": eval_test_split["test"],
    }

    tokenized_datasets = {
        i: raw_datasets[i].map(
            lambda x: preprocess_function(x, tokenizer),
            batched=True,
        ).remove_columns(['source', 'target']) for i in raw_datasets.keys()
    }

    tokenized_datasets['eval'] = tokenized_datasets['eval'].select(range(20)).shuffle(seed=42)

    # Define Data Collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model)

    # initialize metric wer
    wer_metric = load("wer")

    # Set up Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['eval'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Start Training
    trainer.train()
