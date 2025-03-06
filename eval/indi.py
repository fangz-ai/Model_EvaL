import os
import nltk

from abc import ABC
from typing import (Any, Dict, List, Union, Tuple)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from transformers.data.metrics.squad_metrics import compute_f1

from base import segment_words

class Indicators(ABC):
    def __init__(self):
        super().__init__()

    #=====================================================F1-Score=====================================================
    def cal_F1_score(self, answer : List[str], ref : Union[str, List[str]]) -> Dict[str, float]:
        f1 = 0
        if isinstance(ref, list):
            ref_num = len(ref)
            for single_ref in ref:
                f1 = max(f1, compute_f1(single_ref, answer))
        else:
            f1 = compute_f1(ref, answer)

        return {"f1" : f1}

    def compute_f1(self, answer : str, ref : Union[str, List[str]]) -> Dict[str, float]:
        # word_segmentation
        # self.ref_Type_check(ref)
        answer_tokens = segment_words(answer)
        if isinstance(ref, str):
            ref_tokens = segment_words(ref)
        else:
            ref_tokens = [segment_words(r) for r in ref]

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
        
        return {"f1" : F1}

    #=====================================================BLEU=====================================================
    def ngram(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    def cal_BLEU(self, answer : List[str], refs : List[List[str]]) -> Dict[str, float]:

        smooth_fn = SmoothingFunction().method1
        bleu = sentence_bleu(refs, answer, smoothing_function=smooth_fn)

        return {"bleu" : bleu}

    def compute_bleu(self, answer : str, ref : Union[str, List[str]], max_n=4, weights=None) -> Dict[str, float]:

        self.ref_Type_check(ref)
        answer_tokens = segment_words(answer)
        if isinstance(ref, str):
            ref_tokens = segment_words(ref)
        else:
            ref_tokens = [segment_words(r) for r in ref]

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

            precisions.append(match_count / total_count if total_count > 0 else 0)

        if all(p > 0 for p in precisions):
            bleu_score = math.exp(sum(w * math.log(p) for p, w in zip(precisions, weights)))
        else:
            bleu_score = 0
        
        answer_len = len(answer_tokens)
        ref_lens = [len(ref_token) for ref_token in ref_tokens]
        closest_ref_len = min(ref_lens, key=lambda ref_len: (abs(ref_len - answer_len), ref_len))

        if answer_len > closest_ref_len:
            brevity_penalty = 1
        else:
            brevity_penalty = math.exp(1 - closest_ref_len / answer_len) if answer_len > 0 else 0

        return {"bleu" : bleu_score * brevity_penalty}

    #=====================================================ROUGE=====================================================
    def cal_ROUGE(self, answer : str, ref : str) -> Dict[str, float]:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(answer, ref)
        return scores

    def compute_rouge(self, answer : str, ref : Union[str, List[str]]) -> Dict[str, float]:

        # word_segmentation
        # self.ref_Type_check(ref)
        answer_tokens = segment_words(answer)
        if isinstance(ref, str):
            ref_tokens = segment_words(ref)
        else:
            ref_tokens = [segment_words(r) for r in ref]

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



    