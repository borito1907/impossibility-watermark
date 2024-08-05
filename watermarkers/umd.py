import logging
from watermarker import Watermarker

import torch
from transformers import LogitsProcessorList

import textwrap

# UMD
from watermarkers.extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector

log = logging.getLogger(__name__)

def strip_prompt(text, prompt):
    last_word = prompt.split()[-1]
    assistant_marker = f"{last_word}assistant"
    log.info(f"Marker: {assistant_marker}")
    if assistant_marker in text:
        stripped_text = text.split(assistant_marker, 1)[1].strip()
        return stripped_text
    return text

class UMDWatermarker(Watermarker):
    def __init__(self, cfg, pipeline=None, n_attempts=10, **kwargs):
        super().__init__(cfg, pipeline, n_attempts, **kwargs)

    def _setup_watermark_components(self):
        self.watermark_processor = WatermarkLogitsProcessor(
            vocab=list(self.tokenizer.get_vocab().values()),
            gamma=self.cfg.watermark_args.gamma,
            delta=self.cfg.watermark_args.delta,
            seeding_scheme=self.cfg.watermark_args.seeding_scheme
        )
        
        self.watermark_detector = WatermarkDetector(
            tokenizer=self.tokenizer,
            vocab=list(self.tokenizer.get_vocab().values()),
            z_threshold=self.cfg.watermark_args.z_threshold,
            gamma=self.cfg.watermark_args.gamma,
            seeding_scheme=self.cfg.watermark_args.seeding_scheme,
            normalizers=self.cfg.watermark_args.normalizers,
            ignore_repeated_ngrams=self.cfg.watermark_args.ignore_repeated_ngrams,
            device=self.cfg.watermark_args.device,
        )
        
        self.generator_kwargs["logits_processor"] = LogitsProcessorList([self.watermark_processor])

    def generate_watermarked_outputs(self, prompt):
        og_prompt = prompt
        if not self.cfg.is_completion:
            if "Llama" in self.model.config._name_or_path:
                prompt = textwrap.dedent(f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful personal assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""")

        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.cfg.generator_args.max_new_tokens
        ).to(self.model.device)
        outputs = self.model.generate(**inputs, **self.generator_kwargs)

        completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        log.info(f"Completion from UMD: {completion}")
        log.info(f"Prompt: {prompt}")

        # NOTE: Stripping here as well since we change the prompt here.
        if not self.cfg.is_completion:
            completion = strip_prompt(completion, og_prompt)

        log.info(f"Returned Completion: {completion}")
        
        return completion

    def detect(self, completion):
        score = self.watermark_detector.detect(completion)
        score_dict = {key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in score.items()}
        z_score = score_dict['z_score']
        is_detected = score_dict['prediction']
        return is_detected, z_score