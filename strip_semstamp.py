import pandas as pd
from watermarker_factory import get_watermarker
import hydra
import logging
import re
from utils import save_to_csv, get_prompt_or_output, get_prompt_and_id_dev, get_prompt_from_id, count_csv_entries
from omegaconf import OmegaConf
from hydra import initialize, compose

log = logging.getLogger(__name__)

def remove_after_assistant(input_string):
    # Regular expression to find 'assistant' where the previous character isn't a space
    match = re.search(r'(?<! )assistant', input_string)
    if match:
        return input_string[:match.start()]
    return input_string

@hydra.main(version_base=None, config_path="conf", config_name="gen_conf")
def main(cfg):
    folder = f"inputs/dev_semstamp_3"

    new_file_path = f'/local1/borito1907/impossibility-watermark/{folder}/watermarked_texts_assistant.csv'

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
    # cfg.is_completion=True # TODO: CHANGE

    # cfg.watermark_args.only_detect = True



    # folders = [f"semstamp__{partition}" for partition in range(1,8)]
    
    # test_dfs = []

    # for folder in folders:
    #     path = f"/local1/borito1907/impossibility-watermark/inputs/{folder}/watermarked_texts.csv"
    #     df = pd.read_csv(path)
    #     test_dfs.append(df)
    # df = pd.concat(test_dfs, axis=0)

    path = f'/local1/borito1907/impossibility-watermark/{folder}/watermarked_texts.csv'
    df = pd.read_csv(path)
    df = df[df['watermarking_scheme'] == "semstamp_lsh"]

    df['text_stripped'] = df['text'].apply(remove_after_assistant)

    semstamp = get_watermarker(cfg)

    fail_count = 0

    for _, row in df.iterrows():
        stripped_text = row['text_stripped']
        log.info(f"Actual text: {row['text']}")
        log.info(f"Stripped text: {stripped_text}")

        try:
            _, zscore = semstamp.detect(row['text_stripped'])
        except Exception as e:
            log.info(f"Exception: {e}")
            fail_count += 1
            continue

        stats = [{'id': row['id'], 'text': row['text_stripped'], 'zscore' : zscore, 'watermarking_scheme': row['watermarking_scheme'], 'model': row['model'], 'time': row['time']}]
        save_to_csv(stats, new_file_path, rewrite=False)

    log.info(f"Fail Count: {fail_count}")

if __name__ == "__main__":
    main()