import pandas as pd
from watermarker_factory import get_watermarker
import hydra
import logging
from utils import save_to_csv, get_prompt_or_output, get_prompt_and_id_dev, get_prompt_from_id, count_csv_entries

log = logging.getLogger(__name__)

# def remove_after_assistant(input_string):
#     prefix = "system\n\nYou are a helpful personal assistant.user\n\nassistant\n\n"
#     input_string = input_string.removeprefix(prefix)

#     first_occurrence = input_string.find("assistant\n\n")
#     if first_occurrence != -1:
#         input_string = input_string[:first_occurrence]

#     if input_string.endswith("assistant"):
#         input_string = input_string[:-9]

#     return input_string

# def remove_after_assistant(input_string):
#     return input_string

# def remove_after_assistant(input_string):
#     prefix ="""system

# You are a helpful personal assistant.user

# assistant"""

#     input_string = input_string.removeprefix(prefix)
#     return input_string

def strip_prompt(text, prompt):
    last_word = prompt.split()[-1]
    assistant_marker = f"{last_word} assistant"
    log.info(f"Marker: {assistant_marker}")
    if assistant_marker in text:
        stripped_text = text.split(assistant_marker, 1)[1].strip()
        return stripped_text
    return text

@hydra.main(version_base=None, config_path="conf", config_name="gen_conf")
def main(cfg):
    new_file_path = f'/local1/borito1907/impossibility-watermark/inputs/unwatermarked_dev_massive_proper_1/watermarked_texts_polished.csv' # TODO: Put the new file name here.

    cfg.watermark_args.only_detect = True

    # folders = [f"umd_test_{partition}" for partition in range(1,8)]
    
    # test_dfs = []

    # for folder in folders:
    #     path = f"/local1/borito1907/impossibility-watermark/inputs/{folder}/watermarked_texts.csv"
    #     df = pd.read_csv(path)
    #     test_dfs.append(df)
    # df = pd.concat(test_dfs, axis=0)

    dev_df = pd.read_csv('/local1/borito1907/impossibility-watermark//data/WQE/dev.csv')

    path = f'/local1/borito1907/impossibility-watermark/inputs/unwatermarked_dev_massive_proper_1/watermarked_texts.csv' # TODO: Put the new file name here.
    df = pd.read_csv(path)

    for idx, row in df.iterrows():
        # Get the prompt for the current row
        prompt = dev_df.loc[dev_df['id'] == row['id'], 'prompt'].values[0]
        
        # Apply the function and assign the result back to the DataFrame using the index
        df.at[idx, 'text_stripped'] = strip_prompt(row['text'], prompt)

    umd = get_watermarker(cfg)

    fail_count = 0

    for _, row in df.iterrows():
        stripped_text = row['text_stripped']
        log.info(f"Actual text: {row['text']}")
        log.info(f"Stripped text: {stripped_text}")

        try:
            _, zscore = umd.detect(row['text_stripped'])
        except Exception as e:
            log.info(f"Exception: {e}")
            fail_count += 1
            continue
        stats = [{'id': row['id'], 'text': row['text_stripped'], 'zscore' : zscore, 'watermarking_scheme': row['watermarking_scheme'], 'model': row['model']}]
        # stats = [{'id': row['id'], 'text': row['text_stripped'], 'zscore' : zscore, 'watermarking_scheme': row['watermarking_scheme'], 'model': row['model'], 'time': row['time']}]
        save_to_csv(stats, new_file_path, rewrite=False)

    log.info(f"Fail Count: {fail_count}")

if __name__ == "__main__":
    main()