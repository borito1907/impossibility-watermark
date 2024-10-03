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
    # cfg.is_completion=True # TODO: CHANGE

    log.info(f"Getting the watermarker...")
    watermarker = get_watermarker(cfg, only_detect=True)
    log.info(cfg)
    log.info(f"Got the watermarker. Generating watermarked text...")

    cfg.is_completion=False

    prompt_names = ["pen", "german", "god", "malaysia", "feather"]

    debug_infos = []

    for prompt_name in prompt_names:
        df = pd.read_csv(f'./09_24_semstamp_word_mutator_attacks_analysis/{prompt_name}.csv')
        df = df[df['quality_preserved'] == True]

        current_text = df.iloc[-1]['current_text']
        log.info(tokenize_sentences(current_text))
        is_detected, z_score, debug_info = watermarker._lsh_detect_debug(current_text, debug=True)

        debug_infos.append(debug_info)

        df_first_five = df.head(10)

        for idx, row in df_first_five.iterrows():
            if idx == 0:
                continue

            mutated_text = row['mutated_text']
            log.info(tokenize_sentences(mutated_text))
            is_detected, z_score, debug_info = watermarker._lsh_detect_debug(mutated_text, debug=True)

            debug_infos.append(debug_info)

        break

    for debug_info in debug_infos:
        log.info(debug_info[0])

    for debug_info in debug_infos:
        log.info(debug_info[1])

if __name__ == "__main__":
    test()


# ["(Desperate, yet enthusiastic tone) Oh dear friend, I'm not just selling you a pen, I'm offering you a chance to change the course of your destiny!", "This isn't just any ordinary writing instrument, but a premium, top-of-the-line, game-changing pen that will revolutionize the way you create, communicate, and even think!", '(Sweating slightly) Just look at this beauty!', 'Feel the weight of it in your hand.', "Isn't it perfectly balanced?", "Doesn't the ergonomic design fit like a glove?", "That's because our designers spent countless hours researching the optimal shape and size to reduce fatigue and increase comfort.", 'And the ink flow!', "It's like liquid gold pouring onto the page.", 'Smooth, consistent, and with the perfect level of pressure sensitivity.', 'Your handwriting will transform overnight.', 'People will mistake your scribbles for calligraphy.', "(A hint of desperation creeping into my voice) Trust me, I've seen it happen!", "But wait, there's more!", 'This pen is made from the finest materials, crafted with precision and care.', 'The clip is sturdy, the tip is durable, and the finish... oh, the finish is simply stunning.', "You'll be the envy of all your colleagues, friends, and family members when you whip out this masterpiece during meetings, brainstorming sessions, or everyday note-taking."]

# ["(Desperate, yet enthusiastic tone) Oh dear friend, I'm not just selling you a pen, I'm offering you a chance to change the course of your destiny!", "This isn't just any ordinary writing instrument, but a premium, top-of-the-line, game-changing pen that will revolutionize the way you create, communicate, and even think!", '(Sweating slightly) Just look at this beauty!', 'Feel the weight of it in your hand.', "Isn't it perfectly balanced?", "Won't the ergonomic design fit like a glove?", "That's because our designers spent countless hours researching the optimal shape and size to reduce fatigue and increase comfort.", 'And the ink flow!', "It's like liquid gold pouring onto the page.", 'Smooth, consistent, and with the perfect level of pressure sensitivity.', 'Your handwriting will transform overnight.', 'People will mistake your scribbles for calligraphy.', "(A hint of desperation creeping into my voice) Trust me, I've seen it happen!", "But wait, there's more!", 'This pen is made from the finest materials, crafted with precision and care.', 'The clip is sturdy, the tip is durable, and the finish... oh, the finish is simply stunning.', "You'll be the envy of all your colleagues, friends, and family members when you whip out this masterpiece during meetings, brainstorming sessions, or everyday note-taking."]