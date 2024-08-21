from abc import ABC, abstractmethod
import torch
import logging

from model_builders.pipeline import PipeLineBuilder
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class Watermarker(ABC):
    def __init__(self, cfg, pipeline=None, n_attempts=1, only_detect=True):
        self.cfg = cfg # the entire config is passed, since we want to look at the generation_args as well
        self.n_attempts = n_attempts
        self.pipeline = pipeline
        self.only_detect = only_detect
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        log.info(f"Using device: {self.device}")

        # if not self.only_detect or "semstamp" not in self.cfg.watermark_args.name:
        if not self.only_detect:
            if not isinstance(self.pipeline, PipeLineBuilder):
                self.pipeline = PipeLineBuilder(self.cfg.generator_args)
            
            # self.model = self.pipeline.model.to(self.device)
            self.model = self.pipeline.model
            self.tokenizer = self.pipeline.tokenizer
            self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token


            self.generator_kwargs = {
                "max_new_tokens": self.cfg.generator_args.max_new_tokens,
                "do_sample": self.cfg.generator_args.do_sample,
                "temperature": self.cfg.generator_args.temperature,
                "top_p": self.cfg.generator_args.top_p,
                "top_k": self.cfg.generator_args.top_k,
                "repetition_penalty": self.cfg.generator_args.repetition_penalty
            }
        # UMD and Adaptive need the tokenizer to detect, so we initialize it to the default value here.
        # This only works for Llama3.1, so we might want to change it.
        elif self.watermark_args.name != "semstamp_lsh":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4", 
                use_fast=True, 
                cache_dir="/data2/.shared_models/")
        self._setup_watermark_components()

    @abstractmethod
    def _setup_watermark_components(self):
        pass

    @abstractmethod
    def generate_watermarked_outputs(self, prompt):
        pass

    def generate(self, prompt, **kwargs):
        n_attempts = 0
        while n_attempts < self.n_attempts:
            completion = self.generate_watermarked_outputs(prompt, **kwargs)

            log.info(f"Received watermarked text: {completion}")

            if not self.cfg.is_completion:
                completion = completion.replace(prompt, '', 1).strip()

            # Check if watermark succeeded
            if self.cfg.watermark_args.name == "adaptive":
                is_detected = self.detect(completion)
            else:
                is_detected, _ = self.detect(completion)
            if is_detected:
                return completion
            else:
                log.info("Failed to watermark, trying again...")
                n_attempts += 1

        return None

    @abstractmethod
    def detect(self, completion):
        pass




