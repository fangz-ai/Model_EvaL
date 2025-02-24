import os
import argparse
import datasets
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

from ..module_flow import DialogueMapper
from ..language_model.python_demo.pipeline import Model as Bmodel_Base

class Bmodel(Bmodel_Base):
    def __init__(self, args):
        super.__init__()

class Task(ABC):
    VERSION: Optional[Union[int, str]] = None
    DATASET_PATH: Optional[str] = None
    DATASET_NAME: Optional[str] = None

    def __init__(self, args) -> None:
        super().__init__()
        # squad_dataset = datasets.load_dataset("squad", cache_dir="./dataset")
        # squad_dataset = datasets.load_from_disk("./dataset/squad")
        self.squad_dataset = datasets.load_dataset("parquet", data_files="./dataset/squad/plain_text/*.parquet")
        # squad_dataset = datasets.load_dataset(
        #     "parquet",
        #     data_files={
        #         "train": "./dataset/squad/plain_text/train*.parquet",
        #         "validation": "./dataset/squad/plain_text/validation*.parquet"
        #     }
        # )
        self.init_from_args(args)

        self.dialogue_map = DialogueMapper(self.model_type, args.model_path)
        self.tokenizer = self.dialogue_map.tokenizer
        self.EOS = self.dialogue_map.EOS
        self.append_user = self.dialogue_map.append_user
        self.append_assistant = self.dialogue_map.append_assistant
        self.apply_chat_template = self.dialogue_map.apply_chat_template
        self.system_prompt = self.dialogue_map.system_prompt
        self.init_history()
        breakpoint()

    def init_from_args(self, args):
        self.model_path = args.model_path
        self.seq_length = args.seq_length
        self.visual_length = args.visual_length
        self.devid = args.devid
        self.test_mode = args.test_mode
        self.chip = args.chip
    
    def init_history(self):
        self.history = [self.system_prompt]

    def encode_tokens(self):
        self.append_user(self.history, self.input_str)
        text = self.apply_chat_template(self.history)
        tokens = self.tokenizer(text).input_ids
        return tokens

    def format_prompt(self, context, question):
        self.append_user(self.history, context)
        self.append_user(self.history, question)
        return self.encode_tokens()
        
    def evaluate(self, dataset, model):
        self.init_history()
        context = self.squad_dataset['train'][0]['context']
        question = self.squad_dataset['train'][0]['question']
        tokens = format_prompt(context, question)

        if self.test_mode == "bmodel":
            if self.chip == "bm1684x":

        self.stream_answer(tokens)

    # def stream_answer(tokens):
    #     raise NotImplementedError("Subclasses must override this method.")
    @abstractmethod
    def stream_answer(self, tokens):
        pass

    @abstractmethod
    def load_dataset(self, tokens):
        pass

    @abstractmethod
    def update_history(self):
        pass

class BmodelEVA(Task):
    def __init__(self, args):
        super.__init__()
        config_path = os.path.join(args.dir_path, "config.json")
        # config
        with open(config_path, 'r') as file:
            self.config = json.load(file)
        self.model_type = args.model_type if args.model_type else self.config['model_type']

        from language_model.python_demo import chat

        self.model = chat.Model()
        self.init_params(args)
        self.load_model(args.model_path, read_bmodel=True)

    def init_params(args)
        self.model.temperature = args.temperature
        self.model.top_p = args.top_p
        self.model.repeat_penalty = args.repeat_penalty
        self.model.repeat_last_n = args.repeat_last_n
        self.model.max_new_tokens = args.max_new_tokens
        self.model.generation_mode = args.generation_mode
        self.model.embedding_path = os.path.join(args.dir_path, "embedding.bin")
        self.model.NUM_LAYERS = self.config["num_hidden_layers"]
        self.enable_history = args.enable_history
        self.init_history()

    def load_model(self, model_path, read_bmodel):
        self.model.init(self.devices, model_path, read_bmodel)

    def update_history(self):
        if self.model.total_length >= self.model.SEQLEN:
            print("... (reach the maximal length)", flush=True, end="")
            self.init_history()
        else:
            self.append_assistant(self.history, self.answer_cur)

    def stream_answer(self, tokens):
        """
        Stream the answer for the given tokens.
        """
        tok_num = 0
        self.answer_cur = ""
        self.answer_token = []

        # First token
        first_start = time.time()
        token = self.model.forward_first(tokens)
        first_end = time.time()
        full_word_tokens = []
        full_answer = ""
        while token not in self.EOS and self.model.total_length < self.model.SEQLEN:
            full_word_tokens.append(token)
            word = self.tokenizer.decode(full_word_tokens, skip_special_tokens=True)
            if "�" in word:
                token = self.model.forward_next()
                tok_num += 1
                continue
            self.answer_token += full_word_tokens
            full_answer += word
            tok_num += 1
            full_word_tokens = []
            token = self.model.forward_next()

        # counting time
        next_end = time.time()
        first_duration = first_end - first_start
        next_duration = next_end - first_end
        tps = tok_num / next_duration

        if self.enable_history:
            self.answer_cur = self.tokenizer.decode(self.answer_token)
            self.update_history()
        else:
            self.init_history()
        
        return full_answer

def model_evaluate(args: Union[argparse.Namespace, None] = None) -> None:
    if args.test_mode == "bmodel" and args.model_mode == "bmodel":
        task = BmodelEVA(args)
    else:
        raise ValueError("not support now.")


def parse_args():
    # base_args
    parser = argparse.ArgumentParser(description='model_eval', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model_path', type=str, required=True, help='torch or bmodel path, like ./Qwen2-VL-2B-Instruct')
    parser.add_argument('--seq_length', type=int, required=True, help="sequence length")
    parser.add_argument('--visual_length', type=int, default=1024, help="visual length for vision transformer")
    # parser.add_argument('--devid', type=str, type=int, default=0, help="devid")
    parser.add_argument('--model_type', type=str, help="model_type")
    parser.add_argument('--test_mode', type=str, choices=["onnx", "bmodel"], help="test_type")
    parser.add_argument('--chip', type=str, default="bm1684x", choices=["bm1684x", "bm1688", "cv186ah"], help="chip")
    parser.add_argument('--model_mode', type=str, required=True, choices=["torch", "bmodel"], help="tested model's mode")
    
    # model_export_args
    parser.add_argument('--torch_path', type=str, help='torch path, like ./Qwen2-VL-2B-Instruct')
    parser.add_argument('--out_dir', type=str, default='./tmp', help='export onnx/bmodel model to path, defaut is `./tmp`')
    parser.add_argument('--out_bmodel', type=str, default='', help='bmodel name after model_tool --combine')
    # parser.add_argument('--seq_length', type=int, required=True, help="sequence length")
    parser.add_argument('--visual_length', type=int, help="visual length for vision transformer")
    # parser.add_argument('--chip', type=str, default="bm1684x", choices=["bm1684x", "bm1688", "cv186ah"], help="chip")
    parser.add_argument('--quantize', type=str, choices=["bf16", "w8bf16", "w4bf16", "f16", "w8f16", "w4f16"], help="quantize")
    parser.add_argument('--num_device', type=int, default=1, help="num device in compiling bmodel")
    parser.add_argument('--max_workers', type=int, default=3, help="max workers for compiling bmodel in multi-processing")
    parser.add_argument('--tpu_mlir_path', type=str, help="tpu_mlir for compiling bmodel")
    parser.add_argument('--export_type', type=str, choices=["onnx", "bmodel"], default="bmodel", help='export torch/onnx to an onnx/bmodel model')
    parser.add_argument('--debug', type=int, choices=[0, 1], default=0, help='debug mode')

    # pipeline_args
    parser.add_argument('--dir_path', type=str, default="./tmp", help="dir path to the config/embedding/tokenizer")
    parser.add_argument('--bmodel_path', type=str, help='path to the bmodel file')
    parser.add_argument('--devid', type=str, default='0', help='device ID to use')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature scaling factor for the likelihood distribution')
    parser.add_argument('--top_p', type=float, default=1.0, help='cumulative probability of token words to consider as a set of candidates')
    parser.add_argument('--repeat_penalty', type=float, default=1.2, help='penalty for repeated tokens')
    parser.add_argument('--repeat_last_n', type=int, default=32, help='repeat penalty for recent n tokens')
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='max new token length to generate') # 这个参数目前似乎无效？
    parser.add_argument('--generation_mode', type=str, choices=["greedy", "penalty_sample"], default="greedy", help='mode for generating next token')
    parser.add_argument('--enable_history', action='store_true', help="if set, enables storing of history memory")
    # parser.add_argument('--model_type', type=str, help="model type")
    
    args = parser.parse_args()

    if args.model_mode == "torch":
        if args.torch_path is None:
            args.torch_path = args.model_path
        if args.quantize is None:
            raise ValueError("Please provide --quantize if tested model is torch model.")
        if args.export_type != args.test_mode:
            # raise ValueError("Please provide --export_type if tested model is torch model.")
            args.export_type = args.test_mode
            # 这里是否要提供完全的model_export功能？即便不测试bmodel，但允许用户通过model_eva将torch model转为bmodel？
    elif args.model_mode == "bmodel":
        if args.bmodel_path is None:
            args.bmodel_path = args.model_path

    if args.model_mode == "bmodel" and args.test_mode == "onnx":
        raise ValueError("Can not convert model from bmodel to onnx!")

    if args.model_mode == "torch" and args.test_mode == "bmodel":
        args.dir_path = "./tmp"
        # 如果用户只提供torch模型，需要model_eva调用model_export转为bmodel再测试，则tokenizer等所在文件夹的命名默认为tmp

    if args.enable_history == True:
        raise ValueError("not support enable_history now.")

if __name__ == "__main__":
    args = parse_args()
    model_evaluate(args)