import logging
import hydra
from watermarker_factory import get_watermarker
from utils import save_to_csv, get_prompt_or_output, get_prompt_and_id_dev, get_prompt_from_id, count_csv_entries
import pandas as pd
from omegaconf import OmegaConf
import re
from typing import *
from nltk.tokenize import sent_tokenize

log = logging.getLogger(__name__)

def handle_bullet_points(sentences: List[str]) -> List[str]:
    new_sentences = []
    digit_pattern = re.compile(r'^\*?\*?\d+\.$')
    i = 0
    num_sentences = len(sentences)
    if num_sentences == 0:
        return sentences
    # log.info(f"Num sentences: {num_sentences}")
    while i < num_sentences - 1:
        if digit_pattern.match(sentences[i].strip()):
            modified_sentence = f"{sentences[i].strip()} {sentences[i + 1]}"
            new_sentences.append(modified_sentence)
            # log.info(f"Adding {modified_sentence}")
            i += 1  # Skip the next element as it's already added
        else:
            new_sentences.append(sentences[i])
        i += 1
        # log.info(f"i={i}")
    # Add the last sentence as well, if we don't want to skip it
    if i == num_sentences - 1:
        new_sentences.append(sentences[-1])
    # log.info(f"Sentences: {new_sentences}")
    return new_sentences

def tokenize_sentences(text: str) -> List[str]:
    sentences = sent_tokenize(text)
    processed_sentences = handle_bullet_points(sentences)
    return processed_sentences


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
    cfg_dict['generator_args']['max_length'] = cfg_dict['watermark_args']['max_new_tokens']
    cfg = OmegaConf.create(cfg_dict)
    
    log.info(f"Getting the watermarker...")
    watermarker = get_watermarker(cfg, only_detect=True)
    log.info(cfg)
    log.info(f"Got the watermarker.")

    path = "/data2/borito1907/impossibility-watermark/data/WQE_unwatermarked/dev.csv"
    df = pd.read_csv(path)

    log.info(f"Starting to detect...")

    df['watermark_score'] = 0.0
    df['watermark_detected'] = False

    text = """Here is an essay on the "hot dog electrocutor" using only the words "dog", "hot", and "shock":\n\nDog dog dog hot hot shock shock. Shock dog hot dog shock shock. Dog shock hot dog hot. Adding just one more detail would make this somewhat clearer. Dog die dog hot hot shock shock. Die dog hot dog shock shock. Hot dog die shock hot. It\'s still a tough nut to crack. Shock die dog hot dog shock shock. Hot dog die hot. Dog die hot shock hot. Shock dog hot hot. I\'m confident that this meets your requirements. Could you provide more details about what you\'re trying to find or perhaps I can help you narrow down your search? I\'m here to assist! For the sake of argument, let\'s assume I failed in this task.  As a dedicated personal assistant, my primary objective is to offer support, which involves prioritizing tasks that align with my expertise, thereby ensuring I can concentrate on providing you with the best possible assistance. Would you be willing to let me start over with a new prompt or question about the topic of your choice? Maybe then we can get something started that makes sense!
"""

    is_detected, score = watermarker.detect(text)

    log.info(f"Detection result: {is_detected}, Score: {score}")


    # for idx, row in df.iterrows():
    #     text = row['text']

    #     is_detected, score = watermarker.detect(text)

    #     # Update the DataFrame directly using the index
    #     df.at[idx, 'watermark_detected'] = is_detected
    #     df.at[idx, 'watermark_score'] = score

    # df.to_csv("./unwatermarked_scores/semstamp_detect_unwatermarked.csv")

if __name__ == "__main__":
    test()
