# RUN: CUDA_VISIBLE_DEVICES=0 python -m distinguisher.semstamp_watermark
import logging
import hydra
from watermarker_factory import get_watermarker
import os
from utils import save_to_csv, get_prompt_or_output, get_prompt_and_id_dev, get_prompt_from_id, count_csv_entries
from omegaconf import OmegaConf
from hydra import initialize, compose
from tqdm import tqdm
import pandas as pd

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../conf", config_name="gen_conf")
def test(cfg):
    cfg.prompt_file='./distinguisher/entropy_prompts.csv'

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict['watermark_args']['name'] = "semstamp_lsh"
    cfg_dict['watermark_args']['embedder'] = {}
    cfg_dict['watermark_args']['delta'] = 0.01
    cfg_dict['watermark_args']['sp_mode'] = "lsh"
    cfg_dict['watermark_args']['sp_dim'] = 3
    cfg_dict['watermark_args']['lmbd'] = 0.25
    cfg_dict['watermark_args']['max_new_tokens'] = 255
    cfg_dict['watermark_args']['min_new_tokens'] = 245
    cfg_dict['watermark_args']['max_trials'] = 50
    cfg_dict['watermark_args']['critical_max_trials'] = 75
    cfg_dict['watermark_args']['cc_path'] = None
    cfg_dict['watermark_args']['train_data'] = None
    cfg_dict['watermark_args']['device'] = "auto"
    cfg_dict['watermark_args']['len_prompt'] = 32
    cfg_dict['watermark_args']['z_threshold'] = 0.5
    cfg_dict['watermark_args']['use_fine_tuned'] = False
    # cfg_dict['generator_args']['model_name_or_path'] = "facebook/opt-6.7b" # TODO: CHANGE
    cfg_dict['generator_args']['max_length'] = cfg_dict['watermark_args']['max_new_tokens']
    cfg = OmegaConf.create(cfg_dict)
    # cfg.is_completion=True # TODO: CHANGEc
    
    import time
    import textwrap
    
    log.info(f"Getting the watermarker...")
    watermarker = get_watermarker(cfg, only_detect=False)
    log.info(cfg)
    log.info(f"Got the watermarker. Generating watermarked text...")

    stats_file_path='./distinguisher/semstamp_stats.csv'
    watermarked_text_file_path='./distinguisher/semstamp_responses.csv'

    df = pd.read_csv(cfg.prompt_file)
    for _, row in tqdm(df.iterrows(), desc=f'Watermarking with {cfg.watermark_args.name}', total=len(df)):
        entropy, prompt = row['entropy'], row['prompt']
            
        log.info(f"Prompt: {prompt}")
        log.info(f"Entropy Level: {entropy}")

        try:
            for _ in range(3):
                start = time.time()
                watermarked_text = watermarker.generate_watermarked_outputs(prompt, stats_file_path=stats_file_path)
                is_detected, score = watermarker.detect(watermarked_text)
                delta = time.time() - start
                
                log.info(f"Watermarked Text: {watermarked_text}")
                log.info(f"Is Watermark Detected?: {is_detected}")
                log.info(f"Score: {score}")
                log.info(f"Time taken: {delta}")

                stats = [{'prompt': prompt, 'entropy': entropy, 'text': watermarked_text, 'zscore' : score, 'watermarking_scheme': cfg.watermark_args.name, 'model': cfg.generator_args.model_name_or_path, 'time': delta}]
                save_to_csv(stats, watermarked_text_file_path, rewrite=False)
        except Exception as e:
            log.info(f"Exception with Level {entropy}.")
            log.info(f"Exception: {e}")

if __name__ == "__main__":
    test()
