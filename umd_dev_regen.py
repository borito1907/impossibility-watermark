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

    dir_name = f"umd_dev_llama31_massive_regen1"
    base_folder_name = f'./inputs/{dir_name}'

    cfg.prompt_file='./data/WQE/dev.csv'

    cfg.is_completion=False
    
    watermarked_text_file_path=f'{base_folder_name}/watermarked_texts.csv'

    path = f"/local1/borito1907/impossibility-watermark/inputs/umd_dev_massive_1/watermarked_texts.csv"
    df = pd.read_csv(path)
    df = df[df['zscore'] < 3]

    # # Calculate the start and end index based on the partition
    # start_index = (cfg.partition - 1) * 150
    # end_index = start_index + 150

    # # Slice the DataFrame to get only the rows for the current partition
    # df_partition = df.iloc[start_index:end_index]

    for _, row in df.iterrows():
        id = row['id']
        prompt = get_prompt_from_id(cfg.prompt_file, id)
            
        log.info(f"Prompt: {prompt}")
        log.info(f"Prompt ID: {id}")

        try:
            for _ in range(1):
                start = time.time()
                watermarked_text = watermarker.generate(prompt)
                is_detected, score = watermarker.detect(watermarked_text)
                delta = time.time() - start
                
                log.info(f"Watermarked Text: {watermarked_text}")
                log.info(f"Is Watermark Detected?: {is_detected}")
                log.info(f"Score: {score}")
                log.info(f"Time taken: {delta}")

            stats = [{'id': id, 'text': watermarked_text, 'zscore' : score, 'watermarking_scheme': cfg.watermark_args.name, 'model': cfg.generator_args.model_name_or_path, 'time': delta}]
            save_to_csv(stats, watermarked_text_file_path, rewrite=False)
        except Exception as e:
            log.info(f"Exception with Prompt ID {id}.")
            log.info(f"Exception: {e}")

if __name__ == "__main__":
    test()
