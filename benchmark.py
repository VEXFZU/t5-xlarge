from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from evaluate import load
from time import time
from tqdm import tqdm
import json
import argparse

def read_json(f_path):
  with open(f_path, 'r', encoding='utf-8') as file:
      data = json.load(file)

  inputs = [item['question_text'] for item in data]
  targets = [item['answer'] for item in data]
  
  # 비어 있지 않은 데이터만 읽어오기
  valid_data = [(input_text, target) for input_text, target in zip(inputs, targets) if target and target != 'None']
  inputs = [input_text for input_text, _ in valid_data]
  targets = [target for _, target in valid_data]
  
  # Braille blank -> Regular blank
  targets = [elem.replace('⠀', ' ').replace('\u2800', ' ') for elem in targets]
  return inputs, targets

def translate_text(text, model, tokenizer, max_length=256):
    model.to('cuda')
    # 모델의 사전학습 prompt와 일관성 유지
    input_text = f'translate Korean to Braille: {text}\nBraille:'
    inputs = tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True, padding=True).to('cuda')

    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_beams=4,
        early_stopping=True
    )
    output = [token for token in outputs[0] if token not in [0, 1, -100]]
    return output

def eval(preds, targets):
    wer_metric = load("wer")
    cer_metric = load("cer")
    length = len(preds)
    correct, wer, cer = 0, 0, 0
    for i, (pred, target) in enumerate(zip(preds, targets)):
        target = target[:len(pred)]
        pred = pred[:len(target)]

        wer_score = wer_metric.compute(predictions=[pred], references=[target])
        cer_score = cer_metric.compute(predictions=[pred], references=[target])
        wer += wer_score
        cer += cer_score

        if wer_score == 0:
            correct += 1
        else:
            print(f"target: {target}")
            print(f"pred: {pred}")
            print(f"WER Score: {wer_score}")
            print(f"CER Score: {cer_score}")

    print(f"""
Correct: {correct}
Correct Rate: {correct/length},
Avg WER: {wer/length}
Avg CER: {cer/length}
    """)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="SabaPivot/t5-xlarge-ko-kb-2", required=True, help="Path to the model checkpoint.")
parser.add_argument("--revision", type=str, default="", required=False, help="Revision name for the checkpoint.")

args = parser.parse_args()
model_name = args.model_name
revision = args.revision
benchmark_path = "평가지표.json"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision=revision)
tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)

inputs, targets = read_json(benchmark_path)
outputs = []
start = time()
for text in tqdm(inputs, desc="Translating"):
    output = translate_text(text, model, tokenizer)
    outputs.append(output)
print(time() - start)

pred = [tokenizer.decode(output, skip_special_tokens=False) for output in outputs]

eval(pred, targets)
