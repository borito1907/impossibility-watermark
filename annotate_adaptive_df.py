# CUDA_VISIBLE_DEVICES=6,7 python -m annotate_adaptive_df single_arg="Sentence"

# CUDA_VISIBLE_DEVICES=5,6 python -m annotate_adaptive_df  +single_arg="Document2" &> 02_11_doc2step.log

import hydra
import logging
import re
import pandas as pd
import os
import sys
import numpy as np
from watermarker_factory import get_default_watermarker, get_watermarker
from difflib import SequenceMatcher
from omegaconf import OmegaConf
from utils import separate_attacks

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)   

def get_step_size(filename: str):
    if "Document" in filename:
        return 5
    elif "Word" in filename:
        return 50
    elif "Sentence" in filename:
        return 5
    elif "Span" in filename:
        return 10
    
def get_old_step_size(filename: str):
    if "Document" in filename:
        return 10
    elif "Word" in filename:
        return 100
    elif "Sentence" in filename:
        return 15
    elif "Span" in filename:
        return 25

def find_missing_annotated_versions(directory_path='./'):
    # Get all files in the directory
    files = os.listdir(directory_path)

    # Filter for files ending with '_results.csv' or '_results_v2.csv'
    results_files = [f for f in files if f.endswith('_results.csv') or f.endswith('_results_v2.csv')]

    # List to store files without corresponding annotated versions
    missing_annotated_files = []

    # Check for corresponding annotated files
    for result_file in results_files:
        # Handle both '_results.csv' and '_results_v2.csv'
        if result_file.endswith('_results.csv'):
            annotated_file = result_file.replace('_results.csv', '_results_annotated.csv')
        elif result_file.endswith('_results_v2.csv'):
            annotated_file = result_file.replace('_results_v2.csv', '_results_annotated.csv')
        
        # Check if the annotated file exists
        if annotated_file not in files:
            missing_annotated_files.append(result_file)

    return missing_annotated_files

def diff_analysis(response1, response2):
    text1 = re.split(r'(\S+)', response1)
    line1 = text1[1::2]
    text2 = re.split(r'(\S+)', response2)
    line2 = text2[1::2]

    different_word_count = 0
    matcher = SequenceMatcher(None, line1, line2)
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'delete':
            different_word_count += (i2 - i1)  # Words in response1 but not in response2
        elif tag == 'insert':
            different_word_count += (j2 - j1)  # Words in response2 but not in response1
        elif tag == 'replace':
            different_word_count += max(i2 - i1, j2 - j1)  # Replaced words
        
    return different_word_count

@hydra.main(version_base=None, config_path="conf", config_name="gen_conf")
def main(cfg):
    log.info(f"Single command-line argument provided: {cfg.single_arg}")

    dir_path = "/data2/borito1907/sandcastles/attack/traces"

    files = os.listdir(dir_path)
    filenames = [f for f in files if "Adaptive" in f and cfg.single_arg in f and "temp" not in f]

    log.info(f"Files with no annotated versions: {filenames}")

    watermarker = get_default_watermarker("adaptive_neo")

    # Loop over all CSV files in the directory
    for filename in filenames:
        step_size = get_step_size(filename) 
        old_step_size = get_old_step_size(filename)
        
        df_path = os.path.join(dir_path, filename)
        log.info(f"Processing file: {df_path}")

        # Set new_df_path by annotating the filename before .csv
        new_df_path = os.path.join(dir_path, filename.replace('.csv', f'_temp{cfg.second_arg}.csv'))
    
        df = pd.read_csv(df_path) 

        log.info(f"Length of the original big DF: {len(df)}")

        dfs = separate_attacks(df)

        log.info(f"Second Arg: {cfg.second_arg}")
        
        if cfg.second_arg == 1:
            dfs = dfs[:30]
        elif cfg.second_arg == 2:
            dfs = dfs[30:60]
        elif cfg.second_arg == 3:
            dfs = dfs[60:]

        modified_dfs = []

        for df in dfs:
            log.info(f"Length of the DF: {len(df)}")
            prompt = df.iloc[0]['prompt']
            original_text = df.iloc[0]['current_text']
            log.info(f"Prompt: {prompt}")
            log.info(f"Original Text: {original_text}")

            # df['watermark_score'] = np.nan
            # df['watermark_detected'] = np.nan
            df['watermark_score'] = df['watermark_score'].astype(float)
            
            for idx, row in df.iterrows():
                step_num = row['step_num']
                if (step_num % old_step_size == 0):
                    if step_num == 0:
                        text = row['current_text']
                    else:
                        text = row['mutated_text']
                    log.info(f"Initial Watermarked Score of the Row: {row['watermark_score']}")
                    if row['watermark_score'] and pd.isna(row['watermark_score']):
                        log.info(f"Detecting at index {idx}.")
                        log.info(f"Step num is {step_num}.")
                        try:
                            is_detected, zscore = watermarker.detect(text)
                            df.at[idx,'watermark_score'] = zscore
                            log.info(f"Zscore: {zscore}")
                        except Exception as e:
                            log.error(f"Error detecting watermark at index {idx} (step {step_num}): {e}")
                            # Assign default values in case of error
                            is_detected = False
                            zscore = np.nan

            modified_dfs.append(df)

        # Concatenate all the modified DataFrames
        final_df = pd.concat(modified_dfs, ignore_index=True)
        log.info(f"Number of rows in final_df: {len(final_df)}")

        # Save the final concatenated DataFrame to CSV
        final_df.to_csv(new_df_path, mode='w', index=False)

if __name__ == "__main__":
    main()