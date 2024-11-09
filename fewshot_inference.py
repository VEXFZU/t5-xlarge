import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput

def preprocess_few_shot_prompt(korean=None, braille=None):
    # Ensure the inputs are valid
    if not korean or not braille:
        return []

    # Create formatted prompt
    prompts = []
    for i in range(len(korean)):
        prompt = f'translate Korean to Braille: "{korean[i]}"\nBraille: {braille[i]}'
        prompts.append(prompt)

    return prompts

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

# Load the T5 model and tokenizer
model_name = "azaraks/t5-v1.1-large-ko-to-kb"  # or use t5-large for more power if you have the resources
tokenizer = AutoTokenizer.from_pretrained(model_name, revision="v0e5", force_download=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision="v0e5", force_download=True)
model.to('cuda')

# 그 돈으로 집에 ㄴ피아노를 싣고 오신 엊저녁이 아직도 생생하다.
# ⠈⠪⠀⠊⠷⠪⠐⠥⠀⠨⠕⠃⠝⠀⠿⠒⠀⠙⠕⠣⠉⠥⠐⠮⠀⠠⠕⠔⠈⠥⠀⠥⠠⠟⠀⠎⠅⠨⠎⠉⠱⠁⠕⠀⠣⠨⠕⠁⠊⠥⠀⠠⠗⠶⠠⠗⠶⠚⠊⠲
# ko = "그 돈으로 집에 ㄴ피아노를 싣고 오신 엊저녁이 아직도 생생하다."
# print(f"Before: {tokenizer.batch_decode(translate_text(ko))}")

Korean = [
    "① ㄱ, ㄴ ② ㄱ, ㄷ",
    "① ㄱ ② ㄱ, ㄷ",
    "① ㄱ ② ㄷ",
    "①"
]

Braille = [
    "⠼⠂⠀⠿⠁⠐⠀⠿⠒⠀⠼⠆⠀⠿⠁⠐⠀⠿⠔",
    "⠼⠂⠀⠿⠁⠀⠼⠆⠀⠿⠁⠐⠀⠿⠔",
    "⠼⠂⠀⠿⠁⠐⠀⠼⠆⠀⠿⠔",
    "⠼⠂"
]

few_shot_examples = preprocess_few_shot_prompt(Korean, Braille)
target_input = "①, ②"

device = model.device
print(device)
# Encode each few-shot example independently
encoder_outputs = []
for example in few_shot_examples:
    # Tokenize and encode each example
    inputs = tokenizer(example, return_tensors="pt", truncation=True, max_length=128)

    # Move input tensors to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output = model.encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    encoder_outputs.append(output.last_hidden_state)

# Concatenate all encoded few-shot examples
# This step concatenates along the sequence length dimension
concatenated_encoder_outputs = torch.cat(encoder_outputs, dim=1)

# Now, encode the target input
target_inputs = tokenizer(target_input, return_tensors="pt", truncation=True, max_length=128)
target_inputs = {key: value.to(device) for key, value in target_inputs.items()}

target_output = model.encoder(input_ids=target_inputs["input_ids"], attention_mask=target_inputs["attention_mask"])

# Concatenate the target encoded output with the few-shot examples
final_encoder_outputs_tensor  = torch.cat([concatenated_encoder_outputs, target_output.last_hidden_state], dim=1)

# Wrap the tensor in BaseModelOutput
final_encoder_outputs = BaseModelOutput(
    last_hidden_state=final_encoder_outputs_tensor.float()
)

# Prepare the decoder inputs (empty for text generation)
decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]]).to(device)

with torch.no_grad():
    outputs = model.generate(
        encoder_outputs=final_encoder_outputs,  # Pass the structured encoder outputs
        decoder_input_ids=decoder_input_ids,
        max_length=50,
        num_beams=5,
        early_stopping=True
    )

# Decode and print the result
result = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(result)
