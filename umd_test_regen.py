import logging
import hydra
from watermarker_factory import get_watermarker
from utils import save_to_csv, get_prompt_or_output, get_prompt_and_id_dev, get_prompt_from_id, count_csv_entries
import pandas as pd

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="gen_conf")
def test(cfg):
    import time
    import textwrap
    
    log.info(f"Getting the watermarker...")
    watermarker = get_watermarker(cfg, only_detect=False)
    log.info(cfg)
    log.info(f"Got the watermarker. Generating watermarked text...")

    dir_name = f"umd_test_regen_1"
    base_folder_name = f'./inputs/{dir_name}'

    cfg.prompt_file='./data/WQE/test.csv'

    cfg.is_completion=False

    # TODO: Have to log the stats for SemStamp, but not for UMD. This is here so I remember
    # to do it for SemStamp.
    
    watermarked_text_file_path=f'{base_folder_name}/watermarked_texts.csv'

    # TODO: Get the code to load the DF here.
    df = pd.read_csv('blabla')

    for row in df.iterrows():
        if row['zscore'] >= 3:
            continue

        # TODO: Get the prompt and the ID.


        prompt, id = get_prompt_and_id_dev(cfg.prompt_file, prompt_num)
            
        log.info(f"Prompt: {prompt}")
        log.info(f"Prompt ID: {id}")

        for _ in range(1):
            start = time.time()
            watermarked_text = watermarker.generate(prompt)
            is_detected, score = watermarker.detect(watermarked_text)
            delta = time.time() - start
            
            log.info(f"Watermarked Text: {watermarked_text}")
            log.info(f"Is Watermark Detected?: {is_detected}")
            log.info(f"Score: {score}")
            log.info(f"Time taken: {delta}")

        stats = [{'id': id, 'text': watermarked_text, 'zscore' : score, 'watermarking_scheme': cfg.watermark_args.name, 'model': cfg.generator_args.model_name_or_path}]
        save_to_csv(stats, watermarked_text_file_path, rewrite=False)

if __name__ == "__main__":
    test()