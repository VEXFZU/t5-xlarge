import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from data import read_braille_tokens, add_braille_tokens

model = AutoModelForSeq2SeqLM.from_pretrained("/home/careforme.dropout/t5-large/results/241105/checkpoint-1224")
tokenizer = AutoTokenizer.from_pretrained("KETI-AIR/ke-t5-large-ko", legacy=False)
braille_dict = read_braille_tokens()
add_braille_tokens(braille_dict, tokenizer, model)

# Move model to GPU if available
if torch.cuda.is_available():
    model.to("cuda")

def translate_text(text, source_lang="한국어", target_lang="점자", max_length=128):
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

    # Decode the output tokens
    # translated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return outputs

# Test translation
ko = "우주소녀가 유니세프에 기부를 했다."
print(tokenizer.batch_decode(translate_text(ko)))
