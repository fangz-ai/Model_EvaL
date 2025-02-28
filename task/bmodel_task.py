import os
import sys
import json
import time
import argparse
import datasets

from transformers import AutoTokenizer, AutoProcessor
from abc import ABC, abstractmethod

# 获取当前脚本所在目录的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上一层目录的路径
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
eval_dir = os.path.abspath(os.path.join(current_dir, "../eval"))
# 将上一层目录添加到 sys.path
sys.path.insert(0, parent_dir)
sys.path.insert(0, eval_dir)

from detectors import Detectors


# 导入 chat.so
import chat

class BmodelTask():
    def __init__(self, args):
        super().__init__()
        self.tokenizer_path = os.path.join(args.dir_path, "tokenizer")


        # config
        config_path = os.path.join(args.dir_path, "config.json")
        with open(config_path, 'r') as file:
            self.config = json.load(file)

        
        # Initialize model
        self.model_type = args.model_type if args.model_type else self.config['model_type']
        self.model = chat.Model()
        self.init_params(args)

        # Initialize model-specific mapper dynamically
        self.map(args)

        # warm up
        self.tokenizer.decode([0])
        self.init_history

        # load model
        self.load_model(args, read_bmodel=True)

    def init_params(self, args):
        self.devices = [int(d) for d in args.devid.split(",")]
        self.model.temperature = args.temperature
        self.model.top_p = args.top_p
        self.model.repeat_penalty = args.repeat_penalty
        self.model.repeat_last_n = args.repeat_last_n
        self.model.max_new_tokens = args.max_new_tokens
        self.model.generation_mode = args.generation_mode
        self.model.embedding_path = os.path.join(args.dir_path, "embedding.bin")
        self.model.NUM_LAYERS = self.config["num_hidden_layers"]
        self.model.model_type = self.model_type

    def map(self, args):
        """Abstract model-specific mapper into a dictionary."""
        if self.model_type == "qwen2":
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
            self.EOS = [self.tokenizer.eos_token_id]
            self.append_user = lambda history, input_str: history.append(
                {"role": "user", "content": input_str}
            )
            self.append_assistant = lambda history, answer_str: history.append(
                {"role": "assistant", "content": answer_str}
            )
            self.apply_chat_template = lambda history: self.tokenizer.apply_chat_template(
                history, tokenize=False, add_generation_prompt=True
            )
            self.system_prompt = {"role": "system", "content": "You are a helpful assistant."}
        elif self.model_type == "qwen":
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
            self.EOS = [self.tokenizer.im_end_id]
            self.append_user = lambda history, input_str: history.append(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(input_str)
            )
            self.append_assistant = lambda history, answer_str: history.append(answer_str)
            self.apply_chat_template = lambda history: "".join(history)
            self.system_prompt = "<|im_start|>system\nYou are a helpful assistant."
        elif self.model_type == "llama":
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True, use_fast=False)
            self.system_prompt = "<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. " \
                                 "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. " \
                                 "Please ensure that your responses are socially unbiased and positive in nature. " \
                                 "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. " \
                                 "If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n"
            self.EOS = [self.tokenizer.eos_token_id]
            self.append_user = lambda history, input_str: history.append(
                "{} [/INST] ".format(input_str)
            )
            self.append_assistant = lambda history, answer_str: history.append(
                "{} </s><s>[INST] ".format(answer_str)
            )
            self.apply_chat_template = lambda history: "".join(history)
            self.tokenizer.add_prefix_space = False
        elif self.model_type == "llama3":
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
            self.system_prompt = {"role": "system", "content": "You are a helpful assistant."}
            self.EOS = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            self.append_user = lambda history, input_str: history.append(
                {"role": "user", "content": input_str}
            )
            self.append_assistant = lambda history, answer_str: history.append(
                {"role": "assistant", "content": answer_str}
            )
            self.apply_chat_template = lambda history: self.tokenizer.apply_chat_template(
                history, tokenize=False, add_generation_prompt=True
            )
        elif self.model_type == "lwm":
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
            self.system_prompt = "You are a helpful assistant."
            self.EOS = [self.tokenizer.eos_token_id]
            self.append_user = lambda history, input_str: history.append(
                "USER: {} ASSISTANT: ".format(input_str)
            )
            self.append_assistant = lambda history, answer_str: history.append(
                "{} ".format(answer_str)
            )
            self.apply_chat_template = lambda history: "".join(history)
            self.tokenizer.add_prefix_space = False
        else:
            raise NotImplementedError(f"{self.model_type} not support now")

        return


    def load_model(self, args, read_bmodel=True):
        if args.model_path is None:
            raise TypeError("missing argument: model_path")
        
        model_path = args.model_path
        load_start = time.time()
        self.model.init(self.devices, model_path, read_bmodel) # when read_bmodel = false, not to load weight, reuse weight and switch stage_index
        load_end = time.time()
        self.load_time = f"{(load_end - load_start):.3f} s"  
    
    def init_history(self):
        self.history = [self.system_prompt]
    
    def encode_tokens(self):
        self.append_user(self.history, self.input_str)
        text = self.apply_chat_template(self.history)
        tokens = self.tokenizer(text).input_ids
        return tokens

      

    def model_generate(self, question):
        self.init_history()
        self.input_str = question
        tokens = self.encode_tokens()
        token = self.model.forward_first(tokens)

        tok_num = 0
        self.answer_cur = ""
        self.answer_token = []
        next_start = time.time()

        # Following tokens
        full_word_tokens = []
        while token not in self.EOS and self.model.total_length < self.model.SEQLEN:
            self.answer_token.append(token)
            tok_num += 1
            token = self.model.forward_next()

        self.answer_cur = self.tokenizer.decode(self.answer_token)

        # counting time
        next_end = time.time()
        next_duration = next_end - next_start
        tps = tok_num / next_duration

        # evaluate(self.answer_cur, dataset="SQuAD")
        return self.answer_cur
    
    def model_evaluate(self, dataset="SQuAD"):
        # eval = eval.Evaluator()
        detector = Detectors()

        self.squad_dataset = datasets.load_dataset("parquet", data_files="../dataset/squad/plain_text/*.parquet")
        for i in range(10):
            question = self.squad_dataset['train'][i]['context'] + self.squad_dataset['train'][i]['question']
            ref = self.squad_dataset['train'][i]['answers']
            answer = self.model_generate(question)
            languages = detector.test_language_drift(answer)
            print(f"Detected languages: {languages}")
            print("answer: {}, ref: {}".format(answer, ref))
        breakpoint()

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--dir_path", type=str, default="./tmp", help="dir path to the config/embedding/tokenizer")
    parser.add_argument('-b', '--model_path', type=str, default="", help='path to the bmodel file')
    parser.add_argument('-d', '--devid', type=str, default='0', help='device ID to use')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature scaling factor for the likelihood distribution')
    parser.add_argument('--top_p', type=float, default=1.0, help='cumulative probability of token words to consider as a set of candidates')
    parser.add_argument('--repeat_penalty', type=float, default=1.2, help='penalty for repeated tokens')
    parser.add_argument('--repeat_last_n', type=int, default=32, help='repeat penalty for recent n tokens')
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='max new token length to generate')
    parser.add_argument('--generation_mode', type=str, choices=["greedy", "penalty_sample"], default="greedy", help='mode for generating next token')
    parser.add_argument('--model_type', type=str, help="model type")
    args = parser.parse_args()
    bmodel_test = BmodelTask(args)
    answer = bmodel_test.model_evaluate()




    