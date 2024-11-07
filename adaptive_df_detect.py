import logging
import hydra
from watermarker_factory import get_watermarker
import os
from utils import save_to_csv, get_prompt_or_output, get_prompt_and_id_dev, get_prompt_from_id, count_csv_entries
from omegaconf import OmegaConf
from hydra import initialize, compose
import pandas as pd

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="gen_conf")
def test(cfg):
    cfg.prompt_file='./data/WQE/dev.csv'

    # cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    # cfg_dict['generator_args']['top_k'] = 50
    # cfg_dict['generator_args']['top_p'] = 0.9
    # cfg_dict['generator_args']['max_new_tokens'] = 1024 # 285
    # cfg_dict['generator_args']['min_new_tokens'] = 128 # 215
    # cfg_dict['generator_args']['repetition_penalty'] = 1.1
    
    # cfg_dict['watermark_args']['name'] = "adaptive"
    # cfg_dict['watermark_args']['measure_model_name'] = "gpt2-large"
    # cfg_dict['watermark_args']['embedding_model_name'] = "sentence-transformers/all-mpnet-base-v2"
    # cfg_dict['watermark_args']['delta'] = 1.5
    # cfg_dict['watermark_args']['delta_0'] = 1.0
    # cfg_dict['watermark_args']['alpha'] = 2.0
    # cfg_dict['watermark_args']['no_repeat_ngram_size'] = 0
    # cfg_dict['watermark_args']['secret_string'] = 'The quick brown fox jumps over the lazy dog'
    # cfg_dict['watermark_args']['measure_threshold'] = 50
    # cfg_dict['watermark_args']['detection_threshold'] = 95.0
    # cfg_dict['watermark_args']['device'] = 'auto'

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict['generator_args']['top_k'] = 50
    cfg_dict['generator_args']['top_p'] = 0.9
    cfg_dict['generator_args']['max_new_tokens'] = 1024 # 285
    cfg_dict['generator_args']['min_new_tokens'] = 128 # 215
    cfg_dict['generator_args']['repetition_penalty'] = 1.1
    
    cfg_dict['watermark_args']['name'] = "adaptive"
    # cfg_dict['watermark_args']['measure_model_name'] = "gpt2-large"
    cfg_dict['watermark_args']['measure_model_name'] = "EleutherAI/gpt-neo-2.7B"

    cfg_dict['watermark_args']['embedding_model_name'] = "sentence-transformers/all-mpnet-base-v2"
    cfg_dict['watermark_args']['delta'] = 0.25
    cfg_dict['watermark_args']['delta_0'] = 0.0
    cfg_dict['watermark_args']['alpha'] = 4.0
    cfg_dict['watermark_args']['no_repeat_ngram_size'] = 0
    cfg_dict['watermark_args']['secret_string'] = 'The quick brown fox jumps over the lazy dog'
    cfg_dict['watermark_args']['measure_threshold'] = 0
    cfg_dict['watermark_args']['detection_threshold'] = 95.0
    cfg_dict['watermark_args']['device'] = 'auto'

    cfg = OmegaConf.create(cfg_dict)
    
    import time
    import textwrap
    
    log.info(f"Getting the watermarker...")
    watermarker = get_watermarker(cfg, only_detect=True)
    log.info(cfg)
    log.info(f"Got the watermarker.")

    path = "/data2/borito1907/impossibility-watermark/data/WQE_unwatermarked/dev.csv"
    df = pd.read_csv(path)

    log.info(f"Starting to detect...")

    df['watermark_score'] = 0.0
    df['watermark_detected'] = False

    for idx, row in df.iterrows():
        text = row['text']

        is_detected, score = watermarker.detect(text)

        # Update the DataFrame directly using the index
        df.at[idx, 'watermark_detected'] = is_detected
        df.at[idx, 'watermark_score'] = score

        log.info(f"Watermark Detected: {is_detected}")
        log.info(f"Watermark Score: {score}")

    df.to_csv("./unwatermarked_scores/adaptive_detect_unwatermarked_new_params.csv")

if __name__ == "__main__":
    test()
