import os
import nltk
import json
import jieba
import re

from typing import List, Tuple, Union, Dict, Any

def segment_words(text: str) -> Dict[str, Any]:
    # Use regular expressions to replace consecutive whitespace characters 
    # (e.g., spaces, newlines, tabs) with a single space.

    current_dir = os.path.dirname(os.path.abspath(__file__))
    third_party_path = os.path.join(current_dir, '../third_party')
    punkt_path = os.path.join(third_party_path, 'nltk_data/tokenizers/punkt')
    nltk.data.path.append(punkt_path)
    word_tokenize = nltk.tokenize.word_tokenize
    has_illegal_char = False

    config_path = os.path.join(current_dir, '../config/base_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    text = re.sub(r'\s+', ' ', text)

    result = []

    try:
        english_parts = word_tokenize(text)
    except LookupError:
        print("Punkt tokenizer not found. Downloading...")
        try:
            nltk.download('punkt', download_dir=self.third_party_path)
        except:
            nltk.download('punkt_tab', download_dir=self.third_party_path)
        punkt_path = os.path.join(third_party_path, 'punkt')
        nltk.data.path.append(punkt_path)
        word_tokenize = nltk.tokenize.word_tokenize
        english_parts = word_tokenize(text)

    for part in english_parts:
        if part.isascii():
            result.append(part)
        elif any('\u4e00' <= char <= '\u9fff' for char in part):
            chinese_words = jieba.cut(part)
            result.extend(chinese_words)
        elif '\ufffd' in part:
            print(f"Warning: Illegal character detected in part: {part}")
            has_illegal_char = True
            continue
        else:
            result.append(part) 
    
    if has_illegal_char and config.get("allow_illegal_characters", False):
        return {"success": False, "error": "Illegal characters are not allowed."}

    return {"success": True, "result": result}

if __name__ == "__main__":
    text = "我明白了了，小时了了，大未必佳，不了了之了了解了了吗"
    # text = "舟遥遥以轻飏，风飘飘而吹衣"
    words = segment_words(text)
    print(words) 