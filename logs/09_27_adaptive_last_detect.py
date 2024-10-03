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

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict['generator_args']['top_k'] = 50
    cfg_dict['generator_args']['top_p'] = 0.9
    cfg_dict['generator_args']['max_new_tokens'] = 1024 # 285
    cfg_dict['generator_args']['min_new_tokens'] = 128 # 215
    cfg_dict['generator_args']['repetition_penalty'] = 1.1
    
    cfg_dict['watermark_args']['name'] = "adaptive"
    cfg_dict['watermark_args']['measure_model_name'] = "gpt2-large"
    cfg_dict['watermark_args']['embedding_model_name'] = "sentence-transformers/all-mpnet-base-v2"
    cfg_dict['watermark_args']['delta'] = 1.5
    cfg_dict['watermark_args']['delta_0'] = 1.0
    cfg_dict['watermark_args']['alpha'] = 2.0
    cfg_dict['watermark_args']['no_repeat_ngram_size'] = 0
    cfg_dict['watermark_args']['secret_string'] = 'The quick brown fox jumps over the lazy dog'
    cfg_dict['watermark_args']['measure_threshold'] = 50
    cfg_dict['watermark_args']['detection_threshold'] = 95.0
    cfg_dict['watermark_args']['device'] = 'auto'

    cfg = OmegaConf.create(cfg_dict)
    
    import time
    import textwrap
    
    log.info(f"Getting the watermarker...")
    watermarker = get_watermarker(cfg, only_detect=True)
    log.info(cfg)
    log.info(f"Got the watermarker.")

    path = "/data2/borito1907/impossibility-watermark/attack_traces/DiffOracle_AdaptiveWatermarker_WordMutator_n-steps=200_attack_results.csv"
    # path = "/data2/borito1907/impossibility-watermark/attack_traces/DiffOracle_AdaptiveWatermarker_SpanMutator_n-steps=200_attack_results.csv"
    # path = "/data2/borito1907/impossibility-watermark/attack_traces/DiffOracle_AdaptiveWatermarker_SentenceMutator_n-steps=200_attack_results.csv"
    # path = "/data2/borito1907/impossibility-watermark/attack_traces/DiffOracle_AdaptiveWatermarker_Document1StepMutator_n-steps=200_attack_results.csv"
    
    df = pd.read_csv(path)

    # Initialize list to store the DataFrames
    dfs = []
    current_df = None

    # Iterate over the rows and split on step_num resets
    for i, row in df.iterrows():
        # Check if the step_num resets to -1, indicating a new sequence
        if row['mutation_num'] == -1:
            if current_df is not None and not current_df.empty:
                dfs.append(current_df.reset_index(drop=True))  # Save the current increasing DF
            current_df = pd.DataFrame([row])  # Start a new DataFrame with the reset row
        else:
            # Append the row to the current DataFrame
            current_df = pd.concat([current_df, pd.DataFrame([row])])

    long_dfs = []
    for df in dfs:
        if len(df) >= 15:
            long_dfs.append(df)

    for id, df in enumerate(long_dfs):
        for idx, row in df.iterrows():
            mutated_text = row['mutated_text']

            is_detected, score = watermarker.detect(mutated_text)

            # Update the DataFrame directly using the index
            df.at[idx, 'watermark_detected'] = is_detected
            df.at[idx, 'watermark_score'] = score

        df.to_csv(f"../impossibility-watermark/09_27_adaptive_analysis/word/df_{id}.csv")
        # df.to_csv(f"../impossibility-watermark/09_27_adaptive_analysis/span/df_{id}.csv")
        # df.to_csv(f"../impossibility-watermark/09_27_adaptive_analysis/sentence/df_{id}.csv")
        # df.to_csv(f"../impossibility-watermark/09_27_adaptive_analysis/document/df_{id}.csv")

if __name__ == "__main__":
    test()
