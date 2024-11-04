import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from data import read_braille_tokens, add_braille_tokens

model = AutoModelForSeq2SeqLM.from_pretrained("/home/careforme.dropout/braille/results/241104/checkpoint-2800")
tokenizer = AutoTokenizer.from_pretrained("KETI-AIR/ke-t5-large-ko", legacy=False)
braille_dict = read_braille_tokens()
tokenizer = add_braille_tokens(braille_dict, tokenizer, model)

# Move model to GPU if available
if torch.cuda.is_available():
    model.to("cuda")

def translate_text(text, source_lang="한국어", target_lang="점자", max_length=256):
    input_text = f"{source_lang}를 {target_lang}로 변환하세요.\n{source_lang}: {text}\n{target_lang}:"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True)

    # Move inputs to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Generate the translation
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_beams=4,
        early_stopping=True
    )
    print(outputs)

    # Decode the output tokens
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return translated_text

# Test translation
ko = "안전 감지 기능 면에서도 일반 산업용 로봇과 차별화된다. ZKW에 따르면 ‘코봇’은 이동 중에 저항이나 충돌을 감지하면 센서가 해당 정보를 전장설계(PLC) 제어에 전달한다. 그 이후 로봇은 즉시 작업을 중단한다."
print(translate_text(ko))
