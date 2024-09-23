import logging
import hydra
from watermarker_factory import get_watermarker
import os
from utils import save_to_csv, get_prompt_or_output, get_prompt_and_id_dev, get_prompt_from_id, count_csv_entries
from omegaconf import OmegaConf
from hydra import initialize, compose
from model_builders.pipeline import PipeLineBuilder
import torch
from markllm.watermark.auto_watermark import AutoWatermark
from markllm.utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="gen_conf")
def test(cfg):
    import time
    import textwrap
    
    log.info(f"Getting the watermarker...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = PipeLineBuilder(cfg.generator_args)
    
    model = pipeline.model
    tokenizer = pipeline.tokenizer
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    # Transformers config
    transformers_config = TransformersConfig(model=model.to(device),
                                            tokenizer=tokenizer,
                                            vocab_size=tokenizer.vocab_size,
                                            device=device,
                                            max_new_tokens=cfg.generator_args.max_new_tokens,
                                            min_length=cfg.generator_args.min_new_tokens,
                                            do_sample=cfg.generator_args.do_sample,
                                            no_repeat_ngram_size=cfg.generator_args.no_repeat_ngram_size)
    
    # Load watermark algorithm
    myWatermark = AutoWatermark.load('KGW', 
                                    algorithm_config='config/KGW.json',
                                    transformers_config=transformers_config)

    log.info(cfg)
    log.info(f"Got the watermarker. Generating watermarked text...")

    dir_name = f"umd_dev_09_22_markllm_{cfg.partition}"
    base_folder_name = f'./inputs/{dir_name}'
    os.makedirs(os.path.dirname(base_folder_name), exist_ok=True)

    cfg.prompt_file='./data/WQE/dev.csv'

    cfg.is_completion=False
    
    watermarked_text_file_path=f'{base_folder_name}/watermarked_texts.csv'

    start = 1 + (cfg.partition - 1) * 150
    end = 1 + cfg.partition * 150

    for prompt_num in range(start,end):
        prompt, id = get_prompt_and_id_dev(cfg.prompt_file, prompt_num)
            
        log.info(f"Prompt: {prompt}")
        log.info(f"Prompt ID: {id}")

        prompt = textwrap.dedent(f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful personal assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""")

        try:
            for _ in range(1):
                start = time.time()
                watermarked_text = myWatermark.generate_watermarked_text(prompt)
                score = myWatermark.detect_watermark(watermarked_text)
                is_detected = (score >= cfg.watermark_args.z_threshold)
                delta = time.time() - start
                
                log.info(f"Watermarked Text: {watermarked_text}")
                log.info(f"Is Watermark Detected?: {is_detected}")
                log.info(f"Score: {score}")
                log.info(f"Time taken: {delta}")

                stats = [{'id': id, 'text': watermarked_text, 'zscore' : score, 'watermarking_scheme': cfg.watermark_args.name, 'model': cfg.generator_args.model_name_or_path, 'time': delta}]
                save_to_csv(stats, watermarked_text_file_path, rewrite=False)
        except Exception as e:
            log.info(f"Exception with Prompt {prompt_num}.")
            log.info(f"Exception: {e}")

if __name__ == "__main__":
    test()
