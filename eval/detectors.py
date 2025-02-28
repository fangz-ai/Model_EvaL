import os
import re
import fasttext
from abc import ABC
from typing import Dict, List, Tuple, Union
from eval import Evaluator
import nltk
# from nltk.corpus import wordnet

class Detectors(ABC):
    def __init__(self):
        super().__init__()
        self.load_nlp_models()
    
    def load_nlp_models(self):
        """
        Load NLP-related models and resources.

        This function initializes and configures various resources required for NLP tasks, 
        such as WordNet for semantic analysis and FastText for language detection.

        Key functionalities:
        1. Configure the path for third-party resources.
        2. Set up the NLTK WordNet data path to enable semantic features like `synsets`.
        3. Load the FastText language detection model for identifying the language of input text.

        Steps:
        1. Determine the directory of the current script.
        2. Define the path to the third-party resources directory (`../third_party`).
        3. Add the WordNet data path to NLTK's resource paths, ensuring proper functionality of WordNet.
        4. Retrieve the English vocabulary using the NLTK `words` corpus.
        5. Load the FastText language detection model file (`lid.176.bin`) into the `fasttext_model` attribute.

        Dependencies:
        - **NLTK**: Used for WordNet and Words corpus.
        - **FastText**: Used for loading the language detection model.

        Requirements:
        - Ensure the `../third_party/nltk_data` directory contains WordNet and Words data files.
        - Ensure the `../third_party/fasttext/lid.176.bin` file exists; otherwise, the FastText model will fail to load.

        Attributes:
        - `self.synsets`: A reference to the NLTK WordNet `synsets` function for semantic analysis.
        - `self.english_vocab`: A set of English words loaded from the NLTK `words` corpus.
        - `self.fasttext_model`: The loaded FastText language detection model.

        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.third_party_path = os.path.join(current_dir, "../third_party")

        wordnet_path = os.path.join(self.third_party_path, 'nltk_data')
        nltk.data.path.append(wordnet_path)
        self.synsets = nltk.corpus.wordnet.synsets

        words_path = os.path.join(self.third_party_path, 'nltk_data')
        self.english_vocab = set(nltk.corpus.words.words())

        fasttext_path = os.path.join(self.third_party_path, 'fasttext', 'lid.176.bin')
        self.fasttext_model = fasttext.load_model(fasttext_path)

        # from nltk.corpus import wordnet
        # print(wordnet.root)
        # breakpoint()

    def test_language_drift(self, text : str, test_punctuation=False) -> List[str]:
        """
        Detect language drift in the given text by splitting it into language blocks
        and identifying the predominant language in each block.

        Args:
            text (str): The input text to analyze.
            test_punctuation (bool): Whether to include punctuation in the analysis.

        Returns:
            List[str]: A list of detected languages for each segment of the text.
        """
        pattern = re.compile(
            r'[\u4e00-\u9fff]+'  # 中文 Chinese
            r'|[\u3040-\u30ff\u31f0-\u31ff]+'  # 日语 平假名、片假名 Japanese (Hiragana, Katakana)
            r'|[\uac00-\ud7af]+'  # 韩语 Korean
            r'|[\u0400-\u04FF]+'  # 俄语 Russian
            r'|[\u0600-\u06FF]+'  # 阿拉伯语 Arabic
            r'|[\u0370-\u03FF]+'  # 希腊语 Greek
            r'|[\u0E00-\u0E7F]+'  # 泰语 Thai
            r'|[a-zA-Z0-9]+'  # 英文和数字 English and numbers
            r'|[^\w\s]'  # 其他符号 Other symbols
        )
        language_blocks = pattern.findall(text)

        def add_prev_context_before_punctuation(langguage_blocks : List[str], context_window=1) -> List[str]:
            """
            Add surrounding context to punctuation blocks.

            Args:
                blocks (List[str]): The list of language blocks.
                context_window (int): The number of surrounding blocks to include.

            Returns:
                List[str]: The modified list of language blocks.
            """
            result = []
            for i, block in enumerate(language_blocks):
                # Check if the current block is punctuation
                if re.match(r'[^\w\s]', block):
                    if not test_punctuation:
                        continue
                    # Add preceding context to punctuation
                    prev_context = language_blocks[i - context_window] if i - context_window >= 0 else ""
                    # next_context = language_blocks[i + context_window] if i + context_window < len(language_blocks) else ""
                    combined_block = block
                    pre = 0
                    while re.match(r'[^\w\s]', prev_context):
                        combined_block = f"{prev_context}{combined_block}"
                        pre += 1
                        prev_context = language_blocks[i - context_window - pre] if i - context_window - pre >= 0 else ""
                    combined_block = f"{prev_context}{combined_block}"
                    result.append(combined_block)
                else:
                    # Add non-punctuation blocks as is
                    result.append(block)

            return result

        language_blocks = add_prev_context_before_punctuation(language_blocks)
        # Detect language for each block with FastText
        detected_languages = []

        whitelist = {"arms", "amidst"}
        # import nltk
        # # nltk.download('words')
        # from nltk.corpus import words
        # english_vocab = set(words.words())

        from langdetect import detect

        import langid

        usa = "helo"
        print("is_english:", usa.lower() in self.english_vocab)
        print("is_english:", detect(usa))
        language, confidence = langid.classify(usa)
        print(language, confidence)

        for block in language_blocks:
            # fastText 预测语言
            predictions, probabilities = self.fasttext_model.predict(block, k=10)
            if block in whitelist:
                filtered_results = [("en", 1)]
            else:
                filtered_results = [
                    (label.replace("__label__", "").replace("uk", "en"), prob)  # 去掉 "__label__" 前缀
                    for label, prob in zip(predictions, probabilities)
                    if prob > 0.01
                ]

                if not 'en' in filtered_results and re.match(r'[a-zA-Z0-9]+', block):
                    is_english = block.lower() in self.english_vocab
                    if not is_english:
                        definition = self.synsets(block)
                        if definition:
                            is_english = True
                    
                    if is_english:
                        filtered_results.append(("en", 1))

            print(block, filtered_results)
            # language = tuple(prediction.replace("__label__", "") for prediction in predictions) # 去掉标签前缀
            detected_languages.append(filtered_results)
        

        head = 0
        languages = []

        while head < len(detected_languages):
            max_length = 1
            language = detected_languages[head][0][0]
            for head_index in range(len(detected_languages[head])):
                tail = head + 1
                lang = detected_languages[head][head_index][0]
                while tail < len(detected_languages):
                    last = tail
                    for tail_index in range(len(detected_languages[tail])):
                        if detected_languages[tail][tail_index][0] == lang:
                            tail += 1
                            break
                    if tail == last:
                        break

                if (tail - head) > max_length:

                    max_length = tail - head
                    language = detected_languages[head][head_index][0]

            breakpoint()            
            head += max_length
            languages.append(language)
        predict, probability = self.fasttext_model.predict("block.", k=5)

        if len(language) > 1:
            self.language_switch = True
        return languages

    def detect_continuous_repetition_with_hash(self, text : str, k : int) -> List[int]:
        base = 31
        mod = 1e9+7
        n = len(text)

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

                if self.check_adjacent_identical_strings(words, text1):
                    return False

        
        return True

        


    # def text_quality_checker(self, text : str):


# 示例文本
text = "Paris is 法国の首都Paris。こんにちは!"
text = "巴黎是法国的首都！"
text = "The Virgin Mary is said to have allegedly appeared to Saint Bernadette Soubirous in 1858 in Lourdes, France."
text = "The daily student paper at Notre Dame is called \"The Observer\"."
text = "In front of the Notre Dame Main Building, there is a copper statue of Christ with arms upraised, and it bears the legend \"Venite Ad Me Omnes\"."
# text = "In front of the Notre Dame Main Building, there is a copper statue of Christ with arms upraised, and it bears the legend Venite Ad Me Omnes."
# text = "Paris is the caption of France!"
text = "The Grotto at Notre Dame, also known as the Grotto of Lourdes, is a significant religious site located on the campus of the University of Notre Dame in Indiana, USA. Architecturally, it embodies a Catholic character and is designed to resemble the original grotto at Lourdes, France, where the Virgin Mary is reputed to have appeared to Saint Bernadette Soubirous in 1858. This replica serves as a place of prayer and reflection for students, faculty, and visitors, offering a serene and contemplative space amidst the academic environment.\
The Grotto features a statue of the Virgin Mary, often adorned with flowers and candles by visitors, symbolizing devotion and veneration. It is typically surrounded by a garden, which adds to its peaceful atmosphere, encouraging reflection and spiritual connection. The Grotto is often visited by students seeking solace, inspiration, or a moment of quiet reflection, especially during times of personal or academic stress.\
In addition to its spiritual significance, the Grotto also serves as a testament to the university's commitment to Catholic values and traditions, integrating religious symbolism and practices into the daily life of the campus community. It is a physical manifestation of Notre Dame's Catholic character, providing a tangible connection to the faith and its history, particularly the Marian devotion that is central to Catholicism."
text = "这个名为bonjour的国家很美丽"
DET = Detectors()
languages = DET.test_language_drift(text)
print(f"Detected languages: {languages}")
text = "“我明白了了，也懂得了分分合合”"
# text = "山东最高的山是山连山、山连山、山连山"
# text = "channel"
thelist = DET.test_repetition_error(text)
print("是否有字词重复：", thelist)



import nltk
from nltk.corpus import wordnet

# 确保下载了 WordNet 数据
nltk.download('wordnet')

# 示例：查找 "USA" 的定义
print(wordnet.synsets("bonjour"))