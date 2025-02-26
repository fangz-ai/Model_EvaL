import os
import re
import fasttext
from abc import ABC
from typing import Dict, List, Tuple, Union
from eval import Evaluator

class Detectors(ABC):
    def __init__(self):
        super().__init__()
        self.load_model()

    def load_model(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.third_party_path = os.path.join(current_dir, '../third_party')
        fasttext_path = os.path.join(self.third_party_path, 'fasttext', 'lid.176.bin')
        self.model = fasttext.load_model(fasttext_path)

    def test_language_drift(self, text : str) -> List[str]:

        pattern = pattern = re.compile(
            r'[\u4e00-\u9fff]+'  # 中文
            r'|[\u3040-\u30ff\u31f0-\u31ff]+'  # 日语
            r'|[\uac00-\ud7af]+'  # 韩语
            r'|[\u0400-\u04FF]+'  # 俄语
            r'|[\u0600-\u06FF]+'  # 阿拉伯语
            r'|[\u0370-\u03FF]+'  # 希腊语
            r'|[\u0E00-\u0E7F]+'  # 泰语
            r'|[a-zA-Z0-9]+'  # 英文和数字
            r'|[^\w\s]'  # 其他符号
        )
        language_blocks = pattern.findall(text)


        def add_prev_context_before_punctuation(langguage_blocks : List, context_window=1):

            result = []
            for i, block in enumerate(language_blocks):
            # 如果当前块是标点符号（非字母、非数字、非空白字符）
                if re.match(r'[^\w\s]', block):
                    # 获取上下文
                    prev_context = language_blocks[i - context_window] if i - context_window >= 0 else ""
                    # next_context = language_blocks[i + context_window] if i + context_window < len(language_blocks) else ""
                    # 将上下文与标点符号组合
                    combined_block = f"{prev_context}{block}"
                    result.append(combined_block)
                else:
                    # 非标点符号块直接添加
                    result.append(block)

            return result

        language_blocks = add_prev_context_before_punctuation(language_blocks)
        # 用 fastText 检测每个语言块的语言
        detected_languages = []
        for block in language_blocks:
            # fastText 预测语言
            predictions, probabilities = self.model.predict(block, k=5)
            filtered_results = [
                (label.replace("__label__", ""), prob)  # 去掉 "__label__" 前缀
                for label, prob in zip(predictions, probabilities)
                if prob > 0.1
            ]

            # language = tuple(prediction.replace("__label__", "") for prediction in predictions) # 去掉标签前缀
            detected_languages.append(filtered_results)
        

        head = 0
        languages = []

        # breakpoint()
        while head < len(detected_languages):
            max_length = 1
            language = ""
            for head_index in range(len(detected_languages[head])):
                tail = head + 1
                language = detected_languages[head][head_index][0]
                while tail < len(detected_languages):
                    last = tail
                    for tail_index in range(len(detected_languages[tail])):
                        if detected_languages[tail][tail_index][0] == language:
                            tail += 1
                            break
                    if tail == last:
                        break

                if (tail - head) > max_length:
                    max_length = tail - head
                    language = detected_languages[head][head_index][0]
            
            head += max_length

            languages.append(language)
        # breakpoint()
        predict, probability = self.model.predict("block.", k=5)

        if len(language) > 1:
            self.language_switch = True
        return languages

    def detect_continuous_repetition_with_hash(self, text : str, k : int) -> List[int]:
        base = 31
        mod = 1e9+7
        n = len(text)
        # breakpoint()

        if len(text) < 2 * k:
            return []
        
        current_hash = 0
        for i in range(k):
            current_hash = (current_hash * base + ord(text[i])) % mod

        prev_hash = current_hash
        base_k = pow(base, k, int(mod))

        result = []

        for i in range(k, n):
            current_hash = (current_hash * base - ord(text[i-k]) * base_k + ord(text[i])) % mod
            current_hash = (current_hash + mod) % mod

            if i >= 2 * k - 1 and current_hash == prev_hash:
                result.append(i)
            # print(i, i-2*k+1, i-k+1)

            if i >= 2 * k - 1:
                prev_hash = (prev_hash * base - ord(text[i-2*k+1]) * base_k + ord(text[i-k+1])) % mod
                prev_hash = (prev_hash + mod) % mod
        return result

    def calculate_hash(self, s, base=31, mod=10**9+7):
        hash_val = 0
        for char in s:
            hash_val = (hash_val * base + ord(char)) % mod
        return hash_val

    def check_adjacent_identical_strings(self, lst, text):
        base = 31
        mod = 10**9 + 7

        target_hash = self.calculate_hash(text, base, mod)

        n = len(lst)
        for i in range(n - 1):
            hash_i = self.calculate_hash(lst[i], base, mod)
            hash_next = self.calculate_hash(lst[i + 1], base, mod)

        if hash_i == target_hash and hash_next == target_hash:
            if lst[i] == text and lst[i + 1] == text:
                return True

        return False

    def test_repetition_error(self, text : str):
        repeat = dict()
        for i in range(1, len(text)//2):

            i_result = self.detect_continuous_repetition_with_hash(text, i)
            repeat[i] = i_result
        breakpoint()

        # for key, values in repeat:
        #     if values:
        #         for index in values:
        #             text1 = text[index - key + 1 : index]
        #             text2 = text[index - 2 * key + 1 : index - key]

        fenci  = Evaluator()
        words = fenci.segment_words(text)

        for i in range(1, len(text)//2):
            for j in range(len(repeat[i])):
                index = repeat[i][j]
                text1 = text[index - i + 1 : index + 1]
                text2 = text[index - 2 * i + 1 : index - i + 1]
                print(text1, text2)

                if self.check_adjacent_identical_strings(words, text1):
                    return False

        
        return True

        


    # def text_quality_checker(self, text : str):


# 示例文本
text = "Paris is 法国の首都Paris。こんにちは!"
text = "巴黎是法国的首都！"
# text = "Paris is the caption of France!"
DET = Detectors()
languages = DET.test_language_drift(text)
print(f"Detected languages: {languages}")
text = "“我明白了了，也懂得了分分合合”"
# text = "山东最高的山是山连山、山连山、山连山"
# text = "channel"
thelist = DET.test_repetition_error(text)
print("是否有字词重复：", thelist)