import subprocess
import logging
import hydra

from attack import Attack
from utils import get_prompt_or_output

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
log = logging.getLogger(__name__)

import re

def remove_double_triple_commas(text):
    # Remove triple commas first
    text = re.sub(r',,,', ',', text)
    # Then remove double commas
    text = re.sub(r',,', ',', text)
    return text

def run_command(command, filepath):
    log.info(f"Running command: {command}")
    log.info(f"Saving results to {filepath}")

    try:
        with open(filepath, "w") as f:
            # Redirect both stdout and stderr to the same file
            result = subprocess.run(command, shell=True, check=True, text=True, stdout=f, stderr=subprocess.STDOUT)
        log.info("Command executed successfully")
        return "Success", None
    except subprocess.CalledProcessError as e:
        log.error(f"Command failed with error: {e.stderr}")
        # Returning None, stderr if there was an error
        return None, e.stderr

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    # TODO: Adjust the prompt file and prompt num.

    prompt_path = './inputs/mini_c4.csv'
    prompt_num = 6

    prompt = get_prompt_or_output(prompt_path, prompt_num)

    # NOTE: Adjust the prompt accordingly.

    # watermarked_text_path = "./inputs/c4_saves/c4_1_temp_100_divp_15_attempt_1/watermarked_text.csv"
    # watermarked_text_path = "/local1/borito1907/impossibility-watermark/inputs/c4_saves/c4_3_temp_100_divp_20_attempt_1/watermarked_text.csv"
    watermarked_text_path = "inputs/fine_c4_saves/c4_6_temp_100_divp_15_attempt_3/watermarked_text.csv"
    watermarked_text_num = 1

    watermarked_text = get_prompt_or_output(watermarked_text_path, watermarked_text_num)

    # TODO: There should be a better way to do this.
    watermarked_text = remove_double_triple_commas(watermarked_text)

    log.info(f"Prompt: {prompt}")
    log.info(f"Watermarked Text: {watermarked_text}")

    # NOTE: Adjust the log file accordingly.
    cfg.attack_args.log_csv_path = f"./semstamp_attacks/06_18_fine_tuned_c4_sentence_prompt_{prompt_num}.csv"

    attacker = Attack(cfg)
    attacked_text = attacker.attack(prompt, watermarked_text)

    log.info(f"Attacked Response: {attacked_text}")

if __name__ == "__main__":
    main()
