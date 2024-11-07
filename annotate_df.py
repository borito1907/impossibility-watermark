import hydra
import logging
import re
import pandas as pd
import os
from watermarker_factory import get_default_watermarker, get_watermarker
from difflib import SequenceMatcher
from omegaconf import OmegaConf
from hydra import initialize, compose

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)   

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

def extract_score_values(file_path: str) -> list:
    # Regular expression pattern to match the "Watermark Score" float value
    score_pattern = re.compile(r"Score:\s([-+]?\d*\.\d+|\d+)")
    
    score_values = []
    
    # Read the file and extract lines with 'Score:'
    with open(file_path, 'r') as log_file:
        lines = log_file.readlines()

    # Loop through each line and extract the score value
    for line in lines:
        if 'Score:' in line:
            match = score_pattern.search(line)
            if match:
                # Convert the matched score to float and append to the list
                score_values.append(float(match.group(1)))
    
    return score_values

def breakup_attacks(df):
    # Break the DF up into smaller DFs
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

    # Add the last DataFrame if it exists and is non-empty
    if current_df is not None and not current_df.empty:
        dfs.append(current_df.reset_index(drop=True))
    
    return dfs

def breakup_attacks_sandpaper(df):
    # Break the DF up into smaller DFs
    dfs = []
    current_df = None

    # Iterate over the rows and split on step_num resets
    for i, row in df.iterrows():
        # Check if the step_num resets to -1, indicating a new sequence
        if i < len(df) - 1 and df.iloc[i + 1]['step_num'] == 0:
            if current_df is not None and not current_df.empty:
                dfs.append(current_df.reset_index(drop=True))  # Save the current increasing DF
            current_df = pd.DataFrame([row])  # Start a new DataFrame with the reset row
        else:
            # Append the row to the current DataFrame
            current_df = pd.concat([current_df, pd.DataFrame([row])])

    # Add the last DataFrame if it exists and is non-empty
    if current_df is not None and not current_df.empty:
        dfs.append(current_df.reset_index(drop=True))
    
    return dfs

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
    dir_path = "/data2/borito1907/impossibility-watermark/attack_traces"

    filenames = find_missing_annotated_versions(dir_path)

    log.info(f"Files with no annotated versions: {filenames}")

    # Loop over all CSV files in the directory
    for filename in filenames:
        if "umd" in filename or "Entropy" in filename:
            continue
        elif "WeakAdaptive" in filename:
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

            watermarker = get_watermarker(cfg, only_detect=True)
        elif "Adaptive" in filename or "adaptive" in filename:
            watermarker = get_default_watermarker("adaptive")
        elif "SemStamp" in filename:
            watermarker = get_default_watermarker("semstamp")
        else:
            raise Exception("Unknown watermarking scheme.")
        
        if "Document" in filename:
            step_size = 10
        elif "EntropyWord" in filename:
            if "1000" in filename:
                step_size = 100
            else:
                step_size = 20
        elif "Word" in filename:
            step_size = 20
        else:
            step_size = 20
        
        df_path = os.path.join(dir_path, filename)
        log.info(f"Processing file: {df_path}")

        # Set new_df_path by annotating the filename before .csv
        new_df_path = os.path.join(dir_path, filename.replace('.csv', '_annotated.csv'))
    
        df = pd.read_csv(df_path) 

        log.info(f"Length of the original big DF: {len(df)}")

        # Breakup DFs

        if "sandpaper" in filename:
            dfs = breakup_attacks_sandpaper(df)
        else:
            dfs = breakup_attacks(df)

        modified_dfs = []

        for df in dfs:
            if len(df) <= 5:
                log.info(f"The DF is too small, continuing...")
                continue
            log.info(f"Length of the DF: {len(df)}")
            prompt = df.iloc[0]['prompt']
            original_text = df.iloc[0]['current_text']
            current_text = original_text
            log.info(f"Prompt: {prompt}")
            log.info(f"Original Text: {original_text}")

            df['watermark_score'] = df['watermark_score'].astype(float)
            df['diff_length'] = 0  # Initialize a new column for diff_length

            for idx, row in df.iterrows():

                if not pd.isna(row['mutated_text']):
                    diff_length = diff_analysis(current_text, row['mutated_text'])
                    # log.info(f"Diff Length: {diff_length}")
                    df.at[idx, 'diff_length'] = diff_length

                if idx < 5 and "Document" in filename or idx % step_size == 0:
                    is_detected, zscore = watermarker.detect(current_text)
                    df.at[idx, 'watermark_detected'] = is_detected
                    df.at[idx, 'watermark_score'] = zscore
                    log.info(f"Watermark Score: {zscore}")

                if row['quality_preserved']:
                    if pd.isna(row['mutated_text']):
                        log.info(f"Row with NaN in 'mutated_text': {row}")
                        current_text = row['current_text']
                    else:
                        current_text = row['mutated_text']


            is_detected, zscore = watermarker.detect(current_text)
            last_index = df.index[-1]
            df.at[last_index, 'watermark_detected'] = is_detected
            df.at[last_index, 'watermark_score'] = zscore
            log.info(f"Watermark Score: {zscore}")

            log.info(f"Last Row: {df.iloc[-1]}")

            modified_dfs.append(df)
        
        # Concatenate all the modified DataFrames
        final_df = pd.concat(modified_dfs, ignore_index=True)
        log.info(f"Number of rows in final_df: {len(final_df)}")

        # Save the final concatenated DataFrame to CSV
        final_df.to_csv(new_df_path, mode='w', index=False)

if __name__ == "__main__":
    main()