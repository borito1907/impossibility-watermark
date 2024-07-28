import logging
import hydra
from watermarker_factory import get_watermarker
from utils import save_to_csv, get_prompt_or_output, get_prompt_and_id_dev, get_prompt_from_id, count_csv_entries

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="gen_conf")
def test(cfg):
    import time
    import textwrap
    
    log.info(f"Starting to watermark...")

    # Read the prompt and the watermarked text from the input files
    prompt=cfg.prompt

    # NOTE: Changed this to work with dev.csv, use the commented out version for prev CSV files. - Boran
    if not prompt:
        if cfg.prompt_num == -1:
            id = cfg.id
            prompt = get_prompt_from_id(cfg.prompt_file, id)
        else:
            # prompt = get_prompt_or_output(cfg.prompt_file, cfg.prompt_num) 
            prompt, id = get_prompt_and_id_dev(cfg.prompt_file, cfg.prompt_num)
        

    log.info(f"Prompt: {prompt}")
    log.info(f"Prompt ID: {id}")

    log.info(f"Getting the watermarker...")
    watermarker = get_watermarker(cfg, only_detect=False)
    log.info(cfg)
    log.info(f"Got the watermarker. Generating watermarked text...")

    for _ in range(1):
        start = time.time()
        watermarked_text = watermarker.generate(prompt)
        is_detected, score = watermarker.detect(watermarked_text)
        delta = time.time() - start
        
        log.info(f"Watermarked Text: {watermarked_text}")
        log.info(f"Is Watermark Detected?: {is_detected}")
        log.info(f"Score: {score}")
        log.info(f"Time taken: {delta}")

    if cfg.watermarked_text_file_path is not None:
        stats = [{'id': id, 'text': watermarked_text, 'zscore' : score, 'watermarking_scheme': cfg.watermark_args.name, 'model': cfg.generator_args.model_name_or_path}]
        save_to_csv(stats, cfg.watermarked_text_file_path, rewrite=False)

if __name__ == "__main__":
    test()
