import os
import re
import json

from abc import ABC
from typing import List, Tuple, Union, Dict

import nltk
import jieba

from base import segment_words
from detectors import Detectors
from indi import Indicators

class Evaluator(ABC):
    def __init__(self):
        super().__init__()
        self.detect = Detectors()
        self.indicate = Indicators()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, '../config/base_config.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)


    def ref_Type_check(self, ref: Union[str, List[str]]):
        if not isinstance(ref, (str, list)):
            raise ValueError("ref must be either a string or a list of strings!")
        if isinstance(ref, list) and not all(isinstance(item, str) for item in ref):
            raise ValueError("if ref is a list, all elements must be strings!")
    
    def eval(self, answer : str, ref : Union[str, List[str]]):
        """
        Evaluate the answer against the reference.

        Args:
            answer (str): The answer to be evaluated.
            ref (Union[str, List[str]]): The reference, which can be a single string or a list of strings.
        """
        # Check the type of the reference
        self.ref_Type_check(ref)
        print("Model_Output: ", answer)
        print("Reference: ", ref)

        # Segment the reference into tokens
        if isinstance(ref, str):
            ref_tokens = segment_words(ref)
            ref_list = [ref_tokens]
        else:
            ref_list = [segment_words(r) for r in ref]

        answer_tokens = segment_words(answer)

        # Detect language_drift and repetition_error
        answer_languages = self.detect.test_language_drift(answer)

        allowed_languages = set(self.config['allow_languages'])

        unallowed_languages = set(answer_languages) - allowed_languages
        if unallowed_languages:
            raise ValueError(f"Unallowed languages detected: {', '.join(unallowed_languages)}")

        repetition_error = self.detect.test_repetition_error(answer)
        if not repetition_error:
            raise ValueError("Repetition error detected")
    
        # Calculate indicators
        if 'zh' in answer_languages:
            f1 = self.indicate.compute_f1(answer, ref)
            bleu = self.indicate.compute_bleu(answer, ref)
            rouge = self.indicate.compute_rouge(answer, ref)
        else:
            f1 = self.indicate.cal_F1_score(answer, ref)
            bleu = self.indicate.cal_BLEU(answer_tokens, ref_list)
            rouge = self.indicate.cal_ROUGE(answer, ref[0])
        
        print(f1, bleu, rouge, sep='\n')

        return

        

        

        


        
