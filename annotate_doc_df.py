import hydra
import logging
import re
import pandas as pd
import os
from watermarker_factory import get_default_watermarker
from difflib import SequenceMatcher

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

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
    # logfile_path = '/data2/borito1907/impossibility-watermark/10_01_adaptive_annotate_watermark_score.log'
    # score_lines = extract_score_values(logfile_path)

    # watermarking_scheme = "adaptive"
    watermarking_scheme = "semstamp"

    if watermarking_scheme == "semstamp":
        filename_pattern = "SemStamp"
    else:
        filename_pattern = "Adaptive"

    watermarker = get_default_watermarker(watermarking_scheme)

    dir_path = "/data2/borito1907/impossibility-watermark/attack_traces"

    # counter = 0

    # Loop over all CSV files in the directory
    for filename in os.listdir(dir_path):
        # Filter for files with SemStamp or Adaptive watermarkers and n-steps=200
        if (filename_pattern in filename) and "n-steps=200" in filename and filename.endswith("results.csv") and "Document" in filename:

            df_path = os.path.join(dir_path, filename)
            log.info(f"Processing file: {df_path}")

            # Set new_df_path by annotating the filename before .csv
            new_df_path = os.path.join(dir_path, filename.replace('.csv', '_annotatedfirstfive.csv'))
     
            df = pd.read_csv(df_path) 

            log.info(f"Length of the original big DF: {len(df)}")

            # Breakup DFs

            dfs = breakup_attacks(df)

            modified_dfs = []

            for df in dfs:
                success_count = 0
                log.info(f"Length of the DF: {len(df)}")
                prompt = df.iloc[0]['prompt']
                original_text = df.iloc[0]['current_text']
                current_text = original_text
                log.info(f"Prompt: {prompt}")
                log.info(f"Original Text: {original_text}")

                df['watermark_score'] = df['watermark_score'].astype(float)
                df['diff_length'] = 0  # Initialize a new column for diff_length

                quality_preserved = True

                for idx, row in df.iterrows():
                    if not pd.isna(row['mutated_text']):
                        diff_length = diff_analysis(current_text, row['mutated_text'])
                        # log.info(f"Diff Length: {diff_length}")
                        df.at[idx, 'diff_length'] = diff_length

                    # Run the watermark detection
                    if success_count <= 5 and quality_preserved:
                        # log.info(f"Detecting Current Text: {current_text}")
                        # zscore = score_lines[counter]
                        # is_detected = (zscore >= 60.0)
                        is_detected, zscore = watermarker.detect(current_text)
                        df.at[idx, 'watermark_detected'] = is_detected
                        df.at[idx, 'watermark_score'] = zscore
                        log.info(f"Watermark Score: {zscore}")

                        # counter +=1 

                    quality_preserved = row['quality_preserved']
                    if quality_preserved:
                        if pd.isna(row['mutated_text']):
                            log.info(f"Row with NaN in 'mutated_text': {row}")
                            current_text = row['current_text']
                        else:
                            current_text = row['mutated_text']
                            success_count += 1
                    
                    
                modified_dfs.append(df)
            
            # Concatenate all the modified DataFrames
            final_df = pd.concat(modified_dfs, ignore_index=True)
            log.info(f"Number of rows in final_df: {len(final_df)}")

            # Save the final concatenated DataFrame to CSV
            final_df.to_csv(new_df_path, mode='w', index=False)

if __name__ == "__main__":
    main()