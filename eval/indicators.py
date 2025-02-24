import os
import re
import math
import jieba
import nltk

from abc import ABC
from typing import Dict, List, Tuple, Union
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize

class Indicators(ABC):
    def __init__(self):
        super().__init__()
        self.load_nlp_models()
    
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
    
    def ref_Type_check(self, ref: Union[str, List[str]]):
        if not isinstance(ref, (str, list)):
            raise ValueError("ref must be either a string or a list of strings!")
        if isinstance(ref, list) and not all(isinstance(item, str) for item in ref):
            raise ValueError("if ref is a list, all elements must be strings!")

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

        for part in english_parts:
            if part.isascii():
                result.append(part)
            elif any('\u4e00' <= char <= '\u9fff' for char in part):  # 检查是否包含中文字符
                chinese_words = jieba.cut(part)
                result.extend(chinese_words)
            elif '\ufffd' in part:  # 检查是否包含非法字符“�”
                print(f"Warning: Illegal character detected in part: {part}")
                self.has_illegal_char = True
                return result
                continue
            else:
                result.append(part) 

        return result

    def compute_f1(self, answer : str, ref : Union[str, List[str]]) -> float:
        # word_segmentation
        self.ref_Type_check(ref)
        answer_tokens = self.segment_words(answer)
        if isinstance(ref, str):
            ref_tokens = self.segment_words(ref)
        else:
            ref_tokens = [self.segment_words(r) for r in ref]

        # answer_tokens = list(jieba.cut(answer))
        # ref_tokens = [list(jieba.cut(single_ref)) for single_ref in ref]

        F1 = 0
        for ref_token in ref_tokens:
            common_tokens = set(answer_tokens) & set(ref_token)
            correct_token_count = sum(min(answer_tokens.count(token), ref_token.count(token)) for token in common_tokens)

            precision = correct_token_count / len(answer_tokens) if answer_tokens else 0
            recall = correct_token_count / len(ref_token) if ref_token else 0

            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            
            F1 = max(F1, f1)
        
        return F1


    def ngram(self, tokens, n):
        """生成 n-gram 序列"""
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    def compute_bleu(self, answer : str, ref : Union[str, List[str]], max_n=4, weights=None) -> float:
        """
        计算 BLEU 分数。
        Parameters:
            candidate (str): 预测文本。
            references (list of str): 参考文本列表。
            max_n (int): 最大 n-gram 阶数。
            weights (list of float): n-gram 权重。
        Returns:
            float: BLEU 分数。
        """
        
        # word_segmentation
        self.ref_Type_check(ref)
        answer_tokens = self.segment_words(answer)
        if isinstance(ref, str):
            ref_tokens = self.segment_words(ref)
        else:
            ref_tokens = [self.segment_words(r) for r in ref]

        # answer_tokens = list(jieba.cut(answer))
        # ref_tokens = [list(jieba.cut(single_ref)) for single_ref in ref]

        # min_length = min(len(answer_tokens), len(ref_tokens[i]) for i in range(0, len(ref_tokens)-1))
        # min_length = len(answer_tokens)
        # min_length = min(min_length, len(ref_tokens[i]) for i in range(len(ref_tokens)))

        min_length = len(answer_tokens)
        if ref_tokens:
            min_length = min(min_length, *(len(ref_tokens[i]) for i in range(len(ref_tokens))))

        if max_n >= min_length:
            max_n = min_length -1 if min_length > 1 else 1


        if weights is None:
            weights = [1.0 / max_n] * max_n

        precisions = []
        for n in range(1, max_n+1):
            answer_ngrams = Counter(self.ngram(answer_tokens, n))
            max_ref_ngrams = Counter()

            for ref_token in ref_tokens:
                ref_ngrams = Counter(self.ngram(ref_token, n))
                for n_gram in ref_ngrams:
                    max_ref_ngrams[n_gram] = max(max_ref_ngrams[n_gram], ref_ngrams[n_gram])

            match_count = sum(min(count, max_ref_ngrams[ng]) for ng, count in answer_ngrams.items())
            total_count = sum(answer_ngrams.values())

            # breakpoint()
            precisions.append(match_count / total_count if total_count > 0 else 0)

        # 几何平均
        if all(p > 0 for p in precisions):
            bleu_score = math.exp(sum(w * math.log(p) for p, w in zip(precisions, weights)))
        else:
            bleu_score = 0
        
        # 长度惩罚
        answer_len = len(answer_tokens)
        ref_lens = [len(ref_token) for ref_token in ref_tokens]
        closest_ref_len = min(ref_lens, key=lambda ref_len: (abs(ref_len - answer_len), ref_len))

        if answer_len > closest_ref_len:
            brevity_penalty = 1
        else:
            brevity_penalty = math.exp(1 - closest_ref_len / answer_len) if answer_len > 0 else 0

        breakpoint()
        return bleu_score * brevity_penalty

    def compute_rouge(self, answer : str, ref : Union[str, List[str]]) -> Dict[str, float]:
        """
        计算 ROUGE 分数。
        Parameters:
            candidate (str): 预测文本。
            references (list of str): 参考文本列表。
        Returns:
            dict: ROUGE 分数。
        """

        # word_segmentation
        self.ref_Type_check(ref)
        answer_tokens = self.segment_words(answer)
        if isinstance(ref, str):
            ref_tokens = self.segment_words(ref)
        else:
            ref_tokens = [self.segment_words(r) for r in ref]
        # answer_tokens = list(jieba.cut(answer))
        # ref_tokens = [list(jieba.cut(single_ref)) for single_ref in ref]

        # n-gram
        def rouge_n(n):
            answer_ngrams = Counter(self.ngram(answer_tokens, n))
            ref_ngrams = Counter()

            for ref_token in ref_tokens:
                ref_ngrams.update(Counter(self.ngram(ref_token, n)))

            match_count = sum(min(count, ref_ngrams[ng]) for ng, count in answer_ngrams.items())
            total_count = sum(ref_ngrams.values())

            return match_count / total_count if total_count > 0 else 0

        # LCS
        def lcs_length(x, y):
            dp = [[0] * (len(y) + 1) for _ in range(len(x) + 1)]
            for i in range(1, len(x) + 1):
                for j in range(1, len(y) + 1):
                    if x[i - 1] == y[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            
            return dp[-1][-1]

        def rouge_L():
            lcs_scores = []
            for ref_token in ref_tokens:
                lcs = lcs_length(answer_tokens, ref_token)
                recall = lcs / len(ref_token) if len(ref_token) > 0 else 0
                precision = lcs / len(answer_tokens) if len(answer_tokens) > 0 else 0
                f1 = 2 * recall * precision / (recall + precision) if recall + precision > 0 else 0
                lcs_scores.append(f1)

            return max(lcs_scores)
        
        return {
            "ROUGE-1": rouge_n(1),
            "ROUGE-2": rouge_n(2),
            "ROUGE-L": rouge_L()
        }


# 示例
candidate = "这是一个测试句子"
references = ["这是一个测试的句子", "这是一句测试的话"]
# # references = ["这是一个测试句子"]

# bleu_score = compute_bleu(candidate, references)
# print("BLEU 分数:", bleu_score)
# rouge_scores = compute_rouge(candidate, references)
# print("ROUGE 分数:", rouge_scores)

candidate = "法国的首都是巴黎Paris"
references = ["巴黎"]
# references = ["这是一个测试句子"]

INDICATORS_TEST = Indicators()
bleu_score = INDICATORS_TEST.compute_bleu(candidate, references)
print("BLEU 分数:", bleu_score)
rouge_scores = INDICATORS_TEST.compute_rouge(candidate, references)
print("ROUGE 分数:", rouge_scores)
f1_score = INDICATORS_TEST.compute_f1(candidate, references)
print("F1 分数:", f1_score)
