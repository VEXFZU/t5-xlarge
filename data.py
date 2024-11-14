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
    # Returns a list of Braille Unicode characters.
    # All 6-dot Braille patterns are in the range 0x2800 to 0x2840.
    # But we need to use 'space' character instead of Braille Pattern Blank (0x2800)
    # Because the tokenizer will split the text by space.
    # Using Hexadecimal numbers for better understanding of Unicode characters.
    return [chr(i) for i in range(0x2801, 0x2840)]

def load_and_merge_braille_lists(file_path):
    merged_list = []

    # Open the JSON file and load the data
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Iterate over all lists in the JSON and merge them
    for key, value in data.items():
        if isinstance(value, list):  # Ensure the value is a list
            merged_list.extend(value)
    return merged_list


def add_braille_tokens(tokenizer, model):
    braille_list = load_and_merge_braille_lists("special_braille.txt")
    special_tokens_dict = {"additional_special_tokens": read_braille_tokens() + braille_list}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    return tokenizer
