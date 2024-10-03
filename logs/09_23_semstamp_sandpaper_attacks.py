import logging
import hydra
from watermarker_factory import get_watermarker
from utils import save_to_csv, get_prompt_or_output, get_prompt_and_id_dev, get_prompt_from_id, count_csv_entries
import pandas as pd
from omegaconf import OmegaConf

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="gen_conf")
def test(cfg):
    import time
    import textwrap

    cfg.prompt_file='./data/WQE/dev.csv'

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

    log.info(f"Getting the watermarker...")
    watermarker = get_watermarker(cfg, only_detect=True)
    log.info(cfg)
    log.info(f"Got the watermarker. Generating watermarked text...")

    cfg.is_completion=False

    path = "/data2/borito1907/MarkLLM/initial_five_sandpaper_attacks.csv"
    df = pd.read_csv(path)

    log.info(f"Starting to generate...")

    for idx, row in df.iterrows():
        # mutated_text = row['mutated_text']
        mutated_text = row['text']

        is_detected, score = watermarker.detect(mutated_text)

        # Update the DataFrame directly using the index
        df.at[idx, 'watermark_detected'] = is_detected
        df.at[idx, 'watermark_score'] = score

    df.to_csv(f'./09_23_sandpaper_semstamp_attacks/first_five.csv')


if __name__ == "__main__":
    test()
