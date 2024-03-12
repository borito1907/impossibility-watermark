import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
import logging

log = logging.getLogger(__name__)
logging.getLogger('optimum.gptq.quantizer').setLevel(logging.WARNING)

class PipeLineBuilder:
    def __init__(self, cfg):
        self.cfg = cfg
                
        log.info(f"Initializing {cfg.model_name_or_path}")

        # NOTE: Using openai is incompatible with watermarking. 
        if "gpt-" in cfg.model_name_or_path:
            load_dotenv(find_dotenv()) # load openai api key from ./.env
            self.pipeline = ChatOpenAI(model_name=cfg.model_name_or_path)
        
        else: # Using locally hosted model
            if 'gemma' in cfg.model_name_or_path:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                    )

                # Initialize and load the model and tokenizer
                self.model = AutoModelForCausalLM.from_pretrained(
                    cfg.model_name_or_path,
                    revision=cfg.revision,
                    cache_dir=cfg.model_cache_dir,
                    device_map=cfg.device_map,
                    trust_remote_code=cfg.trust_remote_code,
                    quantization_config=quantization_config)           

            else:
                # Initialize and load the model and tokenizer
                self.model = AutoModelForCausalLM.from_pretrained(
                    cfg.model_name_or_path,
                    revision=cfg.revision,
                    cache_dir=cfg.model_cache_dir,
                    device_map=cfg.device_map,
                    trust_remote_code=cfg.trust_remote_code)
             
            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg.model_name_or_path, 
                use_fast=True, 
                cache_dir=cfg.model_cache_dir)

            # Store the pipeline configuration
            self.pipeline_config = {
                "model": self.model,
                "tokenizer": self.tokenizer,
                "max_new_tokens": cfg.max_new_tokens,
                "do_sample": cfg.do_sample,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "top_k": cfg.top_k,
                "repetition_penalty": cfg.repetition_penalty
            }

            # Create the pipeline
            self.pipeline_base = pipeline("text-generation", **self.pipeline_config)
            self.pipeline = HuggingFacePipeline(pipeline=self.pipeline_base)            
    
    def generate_text(self, prompt):
        """
        This function expects a formatted prompt and returns the generated text.
        """
        if "gpt" in self.cfg.model_name_or_path:
            return self.pipeline(prompt)
        return self.pipeline_base(prompt)[0]['generated_text'].replace(prompt, "").strip()
        
    def __call__(self, prompt):
        return self.generate_text(prompt)