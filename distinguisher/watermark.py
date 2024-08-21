# RUN: CUDA_VISIBLE_DEVICES=0,1 python -m distinguisher.watermark
import logging
import hydra
from watermarker_factory import get_watermarker
from utils import save_to_csv
import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../conf", config_name="gen_conf")
def test(cfg):
    import time
    import textwrap
    
    log.info(f"Getting the watermarker...")
    watermarker = get_watermarker(cfg, only_detect=False)
    log.info(cfg)
    log.info(f"Got the watermarker. Generating watermarked text...")

    cfg.prompt_file='./distinguisher/entropy_prompts.csv'
    
    watermarked_text_file_path='./distinguisher/watermarked_responses.csv'

    df = pd.read_csv(cfg.prompt_file)
    for _, row in tqdm(df.iterrows(), desc=f'Watermarking with {cfg.watermark_args.name}', total=len(df)):
        entropy, prompt = row['entropy'], row['prompt']
            
        log.info(f"Prompt: {prompt}")
        log.info(f"Entropy Level: {entropy}")

        try:
            for _ in range(3):
                start = time.time()
                watermarked_text = watermarker.generate_watermarked_outputs(prompt)
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