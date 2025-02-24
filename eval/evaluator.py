import os
import argparse
import datasets
import torch
import spacy
import re
import nltk
import jieba

from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Union,
)
from abc import ABC, abstractmethod
from langdetect import detect
from nltk.stem.porter import PorterStemmer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer

from transformers.data.metrics.squad_metrics import compute_f1

try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class evaluator(ABC):
    def __init__(self):
        super().__init__()

    def cal_F1_score(self, answer, ref):

        answer_tokens = answer.split()
        ref_tokens = ref.split()
        common_tokens = set(answer_tokens) & set(ref_tokens)
        print(set(answer_tokens))
        print(set(ref_tokens))
        print(common_tokens)
        num_common = len(common_tokens)

        if num_common == 0:
            return 0.0
        
        precision = num_common / len(answer)
        recall = num_common / len(ref)

        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def cal_BLEU(self, answer, ref):

        answer_tokens = answer.split()
        ref_tokens = [ref.split()] # ref may include multi items

        smooth_fn = SmoothingFunction().method1

        bleu_score = sentence_bleu(ref_tokens, answer_tokens, smoothing_function=smooth_fn)

        return bleu_score

    def cal_ROUGE(self, answer, ref):

        stemmer=PorterStemmer()
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(ref, answer)

        return rouge_scores

    def detect_language_per_word(self, text):
        
        # tokens = word_tokenize(text)
        # tokens = text.split()

        # doc = nlp(text)
        # tokens = [token.text for token in doc]
        # import stanza

        # stanza.download(lang='multilingual')
        # nlp2 = stanza.Pipeline(lang='multilingual', processor='tokenize')
        # breakpoint()
        # doc = nlp2(text)
        # tokens = [word.text for sent in doc.sentences for word in sent.words]
        text = "Paris is 法国的首都Paris"
        text = "Paris is the caption of France"
        nlp3 = spacy.blank("xx")
        doc = nlp(text)
        tokens = [token.text for token in doc]

        language_tags = []

        import fasttext

        # 下载语言检测模型 lid.176.bin
        model = fasttext.load_model('lid.176.bin')
        for token in tokens:
            try:
                lang = detect(token)
                lang = model.predict(token, k=5)
                breakpoint()
                language_tags.append((token, lang))
            except:
                language_tags.append((token, "unknown"))
        breakpoint()
        
        return language_tags

    def calculate_language_switch_frequency(self, language_tags):

        switches = 0
        for i in range(1, len(language_tags)):
            if language_tags[i][1] != language_tags[i-1][1]:
                switches += 1
        
        return switches

    def fenci(self, text):
        text = re.sub(r'\s+', ' ', text)

        result = []

        try:
            english_parts = word_tokenize(text)
        except:
            nltk.download('punkt_tab')
            english_parts = word_tokenize(text)

        for part in english_parts:
            if part.isascii():
                result.append(part)
            else:
                chinese_words = jieba.cut(part)
                result.extend(chinese_words)
        return result

    def fenci2(self, text):
        import pkuseg

        seg = pkuseg.pkuseg()
        result = seg.cut(text)
        return result

    

EVA_TEST = evaluator()
ref = "The capital of France is Paris."
answer = "巴黎 is the capital of France."

F1_score = EVA_TEST.cal_F1_score(answer, ref)
f1_score = compute_f1(ref, answer)
print(f"F1-Score: {f1_score}")
bleu = EVA_TEST.cal_BLEU(answer, ref)
rouge = EVA_TEST.cal_ROUGE(answer, ref)

print(f"F1 Score: {F1_score}")

print(f"BLEU Score: {bleu}")

print("ROUGE Scores:")
for key, value in rouge.items():
    print(f"{key}: {value}")

text = "黄土高原的最高山是天台山，但天台山是否在陕西呢？天台山在四川，所以we可能不在陕西。 \
        那剩下的shallow就是黄土高原help的其他山了。黄土高原的其他山are包括天台山、大昭山、天台山以南的 \
        plateau、天台山以北的 plateau，以及黄土高原西部的between天台山。其中，天台山是黄土高原的 \
        最高山，但可能在陕西的天台山是否存在呢？或者，我是不是混淆了黄土高原和黄帝陵的位置？\
        黄土高原的天台山else可能在黄土高原的南before部，而天台山以南的 plateau可能位于黄土高原的北部，\
        这样黄土高原的最高山可能在天台山以南的 plateau。"
language_tags = EVA_TEST.detect_language_per_word(text)
switches = EVA_TEST.calculate_language_switch_frequency(language_tags)
print(f"Language switches: {switches}")

seg_result = EVA_TEST.fenci(text)
print(seg_result)

