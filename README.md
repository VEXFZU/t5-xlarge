# T5-XL 모델을 사용한 한국어-점자 번역 모델

This repository contains code for fine-tuning the T5-XLarge model for text generation tasks.

## Setup

### Requirements
- Python 3.10+
- PyTorch 2.11+
- Transformers 4.45.2+
- 40GB+ GPU recommended

Install dependencies:
```
pip install -r requirements.txt
```

## Quick Start

1. Prepare your data in the `data/` directory
2. Configure training parameters in `config.yaml`
3. Run training:
```
python src/train.py --config configs/config.yaml
```

## Project Structure
```
t5-xlarge/
├── data/              # Training and validation data
├── src/               # Source code
│   ├── train.py       # Training script
│   ├── evaluate.py    # Evaluation script
│   └── utils.py       # Helper functions
├── configs/           # Configuration files
├── models/            # Saved model checkpoints
└── README.md
```

## Training

The training script supports:
- Mixed precision training
- Gradient accumulation
- Checkpoint saving
- Wandb logging

Example command:
```
python src/train.py \
    --model_name google/t5-xl-lm-adapt \
    --train_file data/train.json \
    --val_file data/val.json \
    --output_dir models/
```

## Evaluation

Run evaluation on a test set:
```
python src/evaluate.py \
    --model_path models/checkpoint-best \
    --test_file data/test.json
```

## License
MIT

## Contact
For questions or issues, please open a GitHub issue.