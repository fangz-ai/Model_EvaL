import os
import re

from abc import ABC
from typing import List, Tuple, Union, Dict

import nltk
import jieba
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers.data.metrics.squad_metrics import compute_f1
from rouge_score import rouge_scorer

# current_dir = os.path.dirname(os.path.abspath(__file__))
# punkt_path = os.path.join(current_dir, '../third_party', 'punkt')
# nltk.data.find(os.path.join(punkt_path, 'PY3', 'english.pickle'))
# nltk.data.path.append(os.path.join(punkt_path, 'english.pickle'))
# print(punkt_path)
# nltk.data.find(punkt_path)

# # print(nltk.data.find('tokenizers/punkt'))
# text = "Hello! How are you? This is a test."
# print(word_tokenize(text))
# print(nltk.data.path)


class Evaluator(ABC):
    def __init__(self):
        super().__init__()
        self.load_nlp_models()
        self.has_illegal_char = False

    def load_nlp_models(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.third_party_path = os.path.join(current_dir, '../third_party')
        punkt_path = os.path.join(self.third_party_path, 'punkt')
        nltk.data.path.append(punkt_path)
        self.word_tokenize = nltk.tokenize.word_tokenize

        try:
            nltk.data.find(punkt_path)
            nltk.data.find(os.path.join(punkt_path, 'english.pickle'))
            nltk.data.find(os.path.join(punkt_path, 'PY3', 'english.pickle'))
            print(f"Punkt model is loaded from: {punkt_path}")
        except LookupError:
            print("Punkt model not found.")
            
    # word_segmentation
    def segment_words(self, text: str) -> List[str]:
        # Use regular expressions to replace consecutive whitespace characters 
        # (e.g., spaces, newlines, tabs) with a single space.
        text = re.sub(r'\s+', ' ', text)

        result = []

        try:
            english_parts = self.word_tokenize(text)
        except LookupError:
            print("Punkt tokenizer not found. Downloading...")
            try:
                nltk.download('punkt', download_dir=self.third_party_path)
            except:
                nltk.download('punkt_tab', download_dir=self.third_party_path)
            punkt_path = os.path.join(self.third_party_path, 'punkt')
            nltk.data.path.append(punkt_path)
            self.word_tokenize = nltk.tokenize.word_tokenize
            english_parts = self.word_tokenize(text)

        # for part in english_parts:
        #     if part.isascii():
        #         result.append(part)
        #     else:
        #         chinese_words = jieba.cut(part)
        #         result.extend(chinese_words)

        for part in english_parts:
            if part.isascii():
                result.append(part)
            elif any('\u4e00' <= char <= '\u9fff' for char in part):  # 检查是否包含中文字符
                chinese_words = jieba.cut(part)
                result.extend(chinese_words)
            elif '\ufffd' in part:  # 检查是否包含非法字符“�”
                print(f"Warning: Illegal character detected in part: {part}")
                # 根据需求处理非法字符，这里选择跳过该部分
                self.has_illegal_char = True
                return result
                continue
            else:
                result.append(part) 
        print("segment_words: ", result)
        return result

    def ref_Type_check(self, ref: Union[str, List[str]]):
        if not isinstance(ref, (str, list)):
            raise ValueError("ref must be either a string or a list of strings!")
        if isinstance(ref, list) and not all(isinstance(item, str) for item in ref):
            raise ValueError("if ref is a list, all elements must be strings!")
    
    def eval_entrance(self, answer: str, ref: Union[str, List[str]], flatten: bool = False) -> Tuple[List[str], Union[List[str], List[List[str]]]]:
        """
        Evaluates the tokens for the given answer and reference.

        Parameters:
            answer (str): The answer string to be tokenized.
            ref (Union[str, List[str]]): The reference, which can be a string or a list of strings.
            flatten (bool): Whether to return a flattened list of reference tokens.
                            If False, returns a nested list. Default is False.

        Returns:
            Tuple[List[str], Union[List[str], List[List[str]]]]:
                - A list of tokens from the answer.
                - Reference tokens, either as a flattened list or a nested list depending on `flatten`.
        """
        self.ref_Type_check(ref)

        # if isinstance(ref, str):
        #     ref_tokens = self.segment_words(ref)
        # elif isinstance(ref, list):
        #     ref_tokens = []
        #     for ref_part in ref:
        #         if flatten is False:
        #             ref_tokens.append(self.segment_words(ref_part))
        #         else:
        #             ref_tokens.extend(self.segment_words(ref_part))
        # else:
        #     raise ValueError("ref's Type unsupported now!")

        if isinstance(ref, str):
            ref_tokens = self.segment_words(ref)
        else:
            ref_tokens = [self.segment_words(r) for r in ref]
            if flatten:
                ref_tokens = [token for tokens in ref_tokens for token in tokens]

        answer_tokens = self.segment_words(answer)

        return answer_tokens, ref_tokens

    def F1_Score(self, answer: str, ref: Union[str, List[str]]) -> float:
        self.ref_Type_check(ref)

        if isinstance(ref, str):
            f1_score = compute_f1(answer, ref)
            return f1_score
        
        f1_scores = [compute_f1(answer, single_ref) for single_ref in ref]
        return max(f1_scores)

    def BLEU(self, answer: str, ref: Union[str, List[str]]) -> float:
        self.ref_Type_check(ref)

        # answer_tokens = answer.split()
        # ref_tokens = [ref.split()] # ref may include multi items

        answer_tokens, ref_tokens = self.eval_entrance(answer, ref)

        if isinstance(ref_tokens[0], str):
            ref_tokens = [ref_tokens] 

        smooth_fn = SmoothingFunction().method1

        bleu_score = sentence_bleu(ref_tokens, answer_tokens, smoothing_function=smooth_fn)

        return bleu_score

    def ROUGE(self, answer: str, ref: Union[str, List[str]]) -> Dict[str, float]:
        # self.ref_Type_check(ref)

        # # 对答案进行分词，并拼接成字符串
        # answer_tokens = self.segment_words(answer)
        # answer_text = " ".join(answer_tokens)  # 拼接成空格分隔的字符串

        # scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

        # if isinstance(ref, str):
        #     ref_tokens = self.segment_words(ref)
        #     ref_text = " ".join(ref_tokens)
        #     rouge_scores = scorer.score(ref_text, answer_text)
        #     return {key: rouge_scores[key].fmeasure for key in rouge_scores}

        # scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        # for r in ref:
        #     ref_tokens = self.segment_words(r)
        #     ref_text = " ".join(ref_tokens)
        #     rouge_scores = scorer.score(ref_text, answer_text)
        #     for key in scores:
        #         scores[key].append(rouge_scores[key].fmeasure)

        # return {key: sum(values) / len(values) for key, values in scores.items()}

        # answer_tokens, ref_tokens = self.eval_entrance(answer, ref)

        # if isinstance(ref_tokens[0], str):
        #     ref_tokens = [ref_tokens]

        # stemmer = PorterStemmer()
        # scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        # rouge_scores = scorer.score(ref_tokens, answer_tokens)

        # return rouge_scores
        from rouge_score import rouge_scorer, tokenize
        from nltk.stem.porter import PorterStemmer
        stemmer = PorterStemmer()
        answer_tokens = tokenize.tokenize(answer, stemmer=None)
        print("预测文本分词结果:", answer_tokens, "原始输入：", answer)
        ref_tokens = tokenize.tokenize(ref, stemmer=None)

        print("参考文本分词结果:", ref_tokens, "原始输入：", ref)

        self.ref_Type_check(ref)

        if isinstance(ref, list):
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

            for single_ref in ref:
                rouge_scores = scorer.score(single_ref, answer)
                for key in scores:
                    scores[key].append(rouge_scores[key].fmeasure)  # 只取 F-measure 分数

            avg_scores = {key: sum(values) / len(values) for key, values in scores.items()}
            return avg_scores
        else:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge_scores = scorer.score(ref, answer)
            return {key: rouge_scores[key].fmeasure for key in rouge_scores}

    




# EVA_TEST = Evaluator()
# ref = "The capital of France is Paris."
# answer = "巴黎 is the capital of France."
# seg_ref = EVA_TEST.segment_words(ref)
# seg_answer = EVA_TEST.segment_words(answer)
# f1_score = EVA_TEST.F1_Score(answer, ref)
# BLEU = EVA_TEST.BLEU(answer, ref)
# print(f"F1-Score: {f1_score}")
# print(f"seg_ref: {seg_ref}")
# print(f"seg_answer: {seg_answer}")

# text = "黄土高原的最高山是天台山，但天台山是否在陕西呢？天台山在四川，所以we可能不在陕西。 \
#         那剩下的shallow就是黄土高原help的其他山了。黄土高原的其他山are包括天台山、大昭山、天台山以南的 \
#         plateau、天台山以北的 plateau，以及黄土高原西部的between天台山。其中，天台山是黄土高原的 \
#         最高山，但可能在陕西的天台山是否存在呢？或者，我是不是混淆了黄土高原和黄帝陵的位置？\
#         黄土高原的天台山else可能在黄土高原的南before部，而天台山以南的 plateau可能位于黄土高原的北部，\
#         这样黄土高原的最高山可能在天台山以南的 plateau。"

# seg_result = EVA_TEST.segment_words(text)
# print(seg_result)

def main():
    # 实例化 Evaluator 类
    evaluator = Evaluator()

    # 测试数据
    answer = "This is a test sentence."
    ref_single = "This is a test sentence."
    ref_list = [
        "This is a test sentence.",
        "This is a testing sentence.",
        "A test sentence this is."
    ]
    chinese_answer = "这是一个测试句子。"
    chinese_ref = "这是一个测试的句子。"

    # # 测试分词功能
    # print("\n--- Testing Word Segmentation ---")
    # print("Answer tokens:", evaluator.segment_words(answer))
    # print("Chinese answer tokens:", evaluator.segment_words(chinese_answer))

    # # 测试 F1 Score
    # print("\n--- Testing F1 Score ---")
    # print("F1 Score (single ref):", evaluator.F1_Score(answer, ref_single))
    # print("F1 Score (multiple refs):", evaluator.F1_Score(answer, ref_list))

    # # 测试 BLEU Score
    # print("\n--- Testing BLEU Score ---")
    # print("BLEU Score (single ref):", evaluator.BLEU(answer, ref_single))
    # print("BLEU Score (multiple refs):", evaluator.BLEU(answer, ref_list))

    # # 测试 ROUGE Score
    print("\n--- Testing ROUGE Score ---")
    print("ROUGE Score (single ref):", evaluator.ROUGE(answer, ref_single))
    # print("ROUGE Score (multiple refs):", evaluator.ROUGE(answer, ref_list))

    # 测试中文文本的 F1、BLEU 和 ROUGE
    print("\n--- Testing Chinese Text ---")
    print("Chinese F1 Score:", evaluator.F1_Score(chinese_answer, chinese_ref))
    print("Chinese BLEU Score:", evaluator.BLEU(chinese_answer, chinese_ref))
    print("Chinese ROUGE Score:", evaluator.ROUGE(chinese_answer, chinese_ref))


if __name__ == "__main__":
    main()