import logging
import hydra
from watermarker_factory import get_watermarker
import os
from utils import save_to_csv, get_prompt_or_output, get_prompt_and_id_dev, get_prompt_from_id, count_csv_entries
from omegaconf import OmegaConf
from hydra import initialize, compose

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="gen_conf")
def test(cfg):
    cfg.prompt_file='./data/WQE/dev.csv'

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    cfg_dict['generator_args']['model_name_or_path'] = "facebook/opt-6.7b"
    cfg_dict['watermark_args']['name'] = "adaptive"
    cfg_dict['watermark_args']['measure_model_name'] = "gpt2-large"
    cfg_dict['watermark_args']['embedding_model_name'] = "sentence-transformers/all-mpnet-base-v2"
    cfg_dict['watermark_args']['mapping_list'] = {}
    cfg_dict['watermark_args']['alpha'] = 2.0   
    cfg_dict['watermark_args']['top_k'] = 50
    cfg_dict['watermark_args']['top_p'] = 0.9
    cfg_dict['watermark_args']['no_repeat_ngram_size'] = 0
    cfg_dict['watermark_args']['delta_0'] = 1.0
    cfg_dict['watermark_args']['delta'] = 1.5
    cfg_dict['watermark_args']['measure_threshold'] = 50
    cfg_dict['watermark_args']['max_new_tokens'] = 230
    cfg_dict['watermark_args']['min_new_tokens'] = 170
    cfg_dict['watermark_args']['device'] = "auto"
    cfg_dict['watermark_args']['repetition_penality'] = 1.1
    cfg_dict['watermark_args']['secret_string'] = 'The quick brown fox jumps over the lazy dog'
    cfg = OmegaConf.create(cfg_dict)

    cfg.is_completion=False
    
    import time
    import textwrap
    
    log.info(f"Getting the watermarker...")
    watermarker = get_watermarker(cfg, only_detect=False)
    log.info(cfg)
    log.info(f"Got the watermarker. Generating watermarked text...")

    dir_name = f"adaptive_dev_{cfg.partition}"
    base_folder_name = f'./inputs/{dir_name}'
    watermarked_text_file_path=f'{base_folder_name}/watermarked_texts.csv'

    start = 1 + (cfg.partition - 1) * 150
    end = 1 + cfg.partition * 150
    for prompt_num in range(start,end):
        os.makedirs(os.path.dirname(base_folder_name), exist_ok=True)

        prompt, id = get_prompt_and_id_dev(cfg.prompt_file, prompt_num)
            
        log.info(f"Prompt: {prompt}")
        log.info(f"Prompt ID: {id}")

        try:
            for _ in range(1):
                start = time.time()
                watermarked_text = watermarker.generate(prompt)
                is_detected = watermarker.detect(watermarked_text)
                delta = time.time() - start
                
                log.info(f"Watermarked Text: {watermarked_text}")
                log.info(f"Is Watermark Detected?: {is_detected}")
                log.info(f"Time taken: {delta}")
        except Exception as e:
            log.info(f"Exception with Prompt {prompt_num}.")
            log.info(f"Exception: {e}")

        stats = [{'id': id, 'text': watermarked_text, 'watermarking_scheme': cfg.watermark_args.name, 'model': cfg.generator_args.model_name_or_path}]
        save_to_csv(stats, watermarked_text_file_path, rewrite=False)

if __name__ == "__main__":
    test()
