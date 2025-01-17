import logging
import hydra
from watermarker_factory import get_watermarker
import os
from utils import save_to_csv, get_prompt_or_output, get_prompt_and_id_dev, get_prompt_from_id, count_csv_entries
from omegaconf import OmegaConf
from hydra import initialize, compose
import pandas as pd  
import time
import textwrap
from dotenv import load_dotenv, find_dotenv
import tiktoken
import openai
from openai import OpenAI
client = OpenAI()


log = logging.getLogger(__name__)

DEF_MODEL = "gpt-4o"
MODELS = {"gpt-4o": "gpt-4o", "gpt-4": "gpt-4", "gpt-3.5": "gpt-3.5-turbo"}
TOKENIZERS  = {model : tiktoken.encoding_for_model(MODELS[model]) for model in MODELS }
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

@hydra.main(version_base=None, config_path="conf", config_name="gen_conf")
def test(cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    cfg_dict['generator_args']['model_name_or_path'] = "gpt-4o"
    cfg_dict['watermark_args']['name'] = "unwatermarked"

    cfg = OmegaConf.create(cfg_dict)

    dir_name = f"gpt4"
    base_folder_name = f'./experiment_dataset/{dir_name}'
    watermarked_text_file_path=f'{base_folder_name}/watermarked_texts.csv'
    os.makedirs(os.path.dirname(base_folder_name), exist_ok=True)

    model="gpt-4o"

    df = pd.read_csv("/data2/borito1907/impossibility-watermark/distinguisher/ent_prompts.csv")

    for _, row in df.iterrows():
        entropy = row['entropy_level']
        prompt = row['prompt']
        
        log.info(f"Prompt: {prompt}")
        log.info(f"Entropy: {entropy}")

        try:
            for i in range(3):
                start = time.time()

                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {
                            "role": "user",
                            "content": f"{prompt}"
                        }
                    ]
                )
                log.info(f"Completion: {completion}")

                watermarked_text = completion.choices[0].message.content

                delta = time.time() - start
                
                log.info(f"Watermarked Text: {watermarked_text}")
                log.info(f"Time taken: {delta}")

                stats = [{'iteration': i, 'entropy': entropy, 'prompt': prompt, 'text': watermarked_text, 'zscore' : None, 'watermarking_scheme': cfg.watermark_args.name, 'model': model, 'time': delta}]
                save_to_csv(stats, watermarked_text_file_path, rewrite=False)
        except Exception as e:
            log.info(f"Exception with entropy {entropy}.")
            log.info(f"Exception: {e}")

if __name__ == "__main__":
    test()
