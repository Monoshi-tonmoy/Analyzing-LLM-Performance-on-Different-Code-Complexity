from typing import List
import time
import hashlib
import re
from abc import ABC, abstractmethod
from .utils import extract_tag
from ..pt.abst_pt import AbstPt

class AbstLLM(ABC):
    def __init__(self, provider, model):
        self.model = model
        self.provider = provider
        self.MAX_RETRIES = 3
        self.kwargs = {}

        self.ai_config = None
        self.temperature = None
        self.top_p = None
        self.max_tokens = None

        self.prefix_sym = None
        self.suffix_sym = None
        self.mid_sym = None
        self.mask_sym = None

        self.is_init = False
        self.stop = None
    def init_ai_kwargs(self, config):
        self.ai_config = config
        self.temperature = config['temperature']
        self.top_p = config['top_p']
        self.max_tokens = config['max_tokens']
        self.stop = config['stop']
        self.is_init = True

    @abstractmethod
    def chat_llm(self, messages):
        raise NotImplementedError

    @abstractmethod
    def competition_llm(self, prompts):
        raise NotImplementedError

    @abstractmethod
    def extract_text_logprobs(self, model_pred):
        raise NotImplementedError

    def chat_batch(self, pt: AbstPt, tasks: List[dict]):
        preproc = pt.task2msg
        postproc = pt.extract_ans
        query_func = self.chat_llm
        return self.gen_batch(tasks, preproc, postproc, query_func)

    def competition_batch(self, pt: AbstPt, tasks: List[dict]):
        preproc = pt.task2pt
        postproc = pt.extract_ans
        query_func = self.competition_llm
        return self.gen_batch(tasks, preproc, postproc, query_func)

    def gen_batch(self, tasks: List[dict], preproc, postproc, query_func):
        if not self.is_init: raise Exception("Gen Params is Not Init")

        input_list = [preproc(task) for task in tasks]
        t1 = time.time()
        outputs = query_func(input_list)
        t2 = time.time()
        cost_time = t2 - t1

        res = []
        for task, input_v, model_pred in zip(tasks, input_list, outputs):
            if model_pred is not None:
                pred_text, logits = self.extract_text_logprobs(model_pred)
            else:
                pred_text, logits = "", None
            llm_response = {
                'input': input_v,
                "ori_task": task,
                'model_pred': model_pred,
                "ai_config": self.ai_config,
                "pred_text": pred_text,
                "logits": logits,
                'pred_ans': postproc(input_v, pred_text),
                "cost_time": cost_time,
            }
            res.append(llm_response)
        return res