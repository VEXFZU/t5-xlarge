from datasets import Dataset
import glob
import json

def load_data(data_dir):
    src_texts = []
    tgt_texts = []
    json_files = glob.glob(f'{data_dir}/*.json')

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data.get('parallel', []):
                src_texts.append(item['source'])
                tgt_texts.append(item['target'])

    dataset = Dataset.from_dict({"source": src_texts, "target": tgt_texts})
    dataset = dataset.shuffle(seed=42)
    return dataset

def read_braille_tokens():
    return [chr(i) for i in range(0x2801, 0x2840)]

def add_braille_tokens(braille_token, tokenizer, model):
    special_tokens_dict = {"additional_special_tokens": braille_token}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer