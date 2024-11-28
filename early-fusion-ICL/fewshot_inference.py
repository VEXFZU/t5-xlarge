import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput
import time
import argparse

def read_korean_braille_pairs(file_path):
    """
    txt file format:
    ① ㄱ, ㄴ ② ㄱ, ㄷ
    ⠼⠂⠀⠿⠁⠐⠀⠿⠒⠀⠼⠆⠀⠿⠁⠐⠀⠿⠔

    ① ㄱ ② ㄱ, ㄷ
    ⠼⠂⠀⠿⠁⠀⠼⠆⠀⠿⠁⠐⠀⠿⠔

    ① ㄱ ② ㄷ
    ⠼⠂⠀⠿⠁⠐⠀⠼⠆⠀⠿⠔
    """
    korean_list = []
    braille_list = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file if line.strip()]
            for i in range(0, len(lines), 2):
                korean = lines[i]
                braille = lines[i + 1]
                korean_list.append(korean)
                braille_list.append(braille)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except IndexError:
        print("Error: File format is incorrect. Ensure each Korean line is followed by its Braille counterpart.")

    return korean_list, braille_list

def preprocess_few_shot_prompt(korean=None, braille=None):
    if not korean or not braille:
        return []

    prompts = []
    for i in range(len(korean)):
        prompt = f'translate Korean to Braille: "{korean[i]}"\nBraille: {braille[i]}'
        prompts.append(prompt)

    return prompts

def translate_text(text, source_lang="한국어", target_lang="점자", max_length=256):
    input_text = f'translate Korean to Braille: {text}\nBraille:'
    inputs = tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True)

    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Generate the translation
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_beams=4,
        early_stopping=True
    )
    return outputs


def few_shot_inference(model, tokenizer, examples, target_input):
    # Early-fusion 구현
    device = model.device

    encoder_outputs = []
    for example in examples:
        inputs = tokenizer(example, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            output = model.encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        encoder_outputs.append(output.last_hidden_state)

    concatenated_encoder_outputs = torch.cat(encoder_outputs, dim=1)

    # Encoding target_inputs
    target_inputs = tokenizer(target_input, return_tensors="pt", truncation=True, max_length=512).to(device)

    target_output = model.encoder(input_ids=target_inputs["input_ids"], attention_mask=target_inputs["attention_mask"])

    # Concat 순서에 따라서 최종 결과에 영향을 끼칠 수 있음. target_output을 먼저 할당하기를 권장
    final_encoder_outputs_tensor = torch.cat([target_output.last_hidden_state, concatenated_encoder_outputs], dim=1)

    # T5 encoder output = decoder input 형식으로 변환
    final_encoder_outputs = BaseModelOutput(
        last_hidden_state=final_encoder_outputs_tensor.float()
    )

    # Decoding
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]]).to(device)

    with torch.no_grad():
        outputs = model.generate(
            encoder_outputs=final_encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            max_length=50,
            num_beams=5,
            early_stopping=True
        )

    # 최종 결과 출력
    result = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return result

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="VEXFZU/t5-xlarge-ko-kb", required=True, help="Path to the model checkpoint.")
parser.add_argument("--revision", type=str, default="main", required=False, help="Revision name for the checkpoint.")
parser.add_argument("--fewshot_path", type=str, default="fewshot-examples.txt", required=True, help="Path to the fewshot examples file.")
parser.add_argument("--target_input", type=str, default="삶과 죽음을 넘나드는 험난한 인생에서", required=True, help="Target input to translate.")
args = parser.parse_args()

model_name = args.model_name
fewshot_path = args.fewshot_path
target_input = args.target_input

model = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision=revision).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)

# Load fewshot
Korean, Braille = read_korean_braille_pairs(fewshot_path)
few_shot_examples = preprocess_few_shot_prompt(Korean, Braille)

# Fewshot inference
result_fewshot = few_shot_inference(model, tokenizer, few_shot_examples, target_input)

# zero shot
result_zeroshot = tokenizer.batch_decode(translate_text(target_input), skip_special_tokens=False)

print(f"Early-fusion fewshot: {result_fewshot}")
print(f"Zeroshot: {result_zeroshot}")