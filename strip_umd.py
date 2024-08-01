import pandas as pd
from watermarker_factory import get_watermarker
import hydra
import logging
from utils import save_to_csv, get_prompt_or_output, get_prompt_and_id_dev, get_prompt_from_id, count_csv_entries

log = logging.getLogger(__name__)

def remove_after_assistant(input_string):
    if input_string.startswith("system"):
        # Find the second occurrence of "assistant"
        first_occurrence = input_string.find("assistant")
        if first_occurrence != -1:
            second_occurrence = input_string.find("assistant", first_occurrence + 1)
            if second_occurrence != -1:
                return input_string[:second_occurrence]
            else:
                return input_string[:first_occurrence]
    else:
        # Find the first occurrence of "assistant"
        first_occurrence = input_string.find("assistant")
        if first_occurrence != -1:
            return input_string[:first_occurrence]
    
    return input_string

@hydra.main(version_base=None, config_path="conf", config_name="gen_conf")
def main(cfg):
    # file_path = '/local1/borito1907/impossibility-watermark/inputs/dev_umd_long_3/watermarked_texts.csv'
    new_file_path = '/local1/borito1907/impossibility-watermark/inputs/test_umd_draft/stripped_watermarked_texts.csv'

    cfg.watermark_args.only_detect = True

    # df = pd.read_csv(file_path)


    test_1 = pd.read_csv('/local1/borito1907/impossibility-watermark/inputs/umd_test_1/watermarked_texts.csv')
    test_2 = pd.read_csv('/local1/borito1907/impossibility-watermark/inputs/umd_test_2/watermarked_texts.csv')
    test_3 = pd.read_csv('/local1/borito1907/impossibility-watermark/inputs/umd_test_3/watermarked_texts.csv')
    test_4 = pd.read_csv('/local1/borito1907/impossibility-watermark/inputs/umd_test_4/watermarked_texts.csv')
    test_dfs = [test_1,test_2,test_3,test_4]
    df = pd.concat(test_dfs, axis=0)

    df['text_stripped'] = df['text'].apply(remove_after_assistant)


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
        save_to_csv(stats, new_file_path, rewrite=False)

    log.info(f"Fail Count: {fail_count}")

if __name__ == "__main__":
    main()