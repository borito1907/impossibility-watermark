import logging
import hydra
from watermarker_factory import get_watermarker
import os
from utils import save_to_csv, get_prompt_or_output, get_prompt_and_id_dev, get_prompt_from_id, count_csv_entries
from omegaconf import OmegaConf
from hydra import initialize, compose
from model_builders.pipeline import PipeLineBuilder
import textwrap

log = logging.getLogger(__name__)

def strip_prompt(text, prompt):
    last_word = prompt.split()[-1]
    assistant_marker = f"{last_word}assistant"
    log.info(f"Marker: {assistant_marker}")
    if assistant_marker in text:
        stripped_text = text.split(assistant_marker, 1)[1].strip()
        return stripped_text
    return text

def generate(prompt, cfg, model , tokenizer, **generator_kwargs):
    og_prompt = prompt
    if not cfg.is_completion:
        if "Llama" in model.config._name_or_path:
            prompt = textwrap.dedent(f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful personal assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""")

    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=cfg.generator_args.max_new_tokens
    ).to(model.device)
    outputs = model.generate(**inputs, **generator_kwargs)

    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)

    log.info(f"Completion: {completion}")
    log.info(f"Prompt: {prompt}")

    # NOTE: Stripping here as well since we change the prompt here.
    if not cfg.is_completion:
        completion = strip_prompt(completion, og_prompt)

    log.info(f"Returned Completion: {completion}")
    
    return completion

@hydra.main(version_base=None, config_path="conf", config_name="gen_conf")
def test(cfg):
    cfg.prompt_file='./data/WQE/dev.csv'

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict['generator_args']['top_k'] = 50
    cfg_dict['generator_args']['top_p'] = 0.9
    cfg_dict['generator_args']['repetition_penalty'] = 1.1
    cfg_dict['generator_args']['min_new_tokens'] = 128 # 215
    cfg_dict['generator_args']['max_new_tokens'] = 1024 # 215
    cfg_dict['generator_args']['temperature'] = 1.0
    cfg_dict['generator_args']['do_sample'] = True
    cfg_dict['generator_args']['device_map'] = 'auto'
    cfg_dict['generator_args']['model_name_or_path'] = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
    cfg_dict['is_completion'] = False

    cfg = OmegaConf.create(cfg_dict)
    
    import time
    import textwrap

    dir_name = f"unwatermarked_dev_massive_proper_{cfg.partition}"
    base_folder_name = f'./inputs/{dir_name}'
    os.makedirs(os.path.dirname(base_folder_name), exist_ok=True)

    watermarked_text_file_path=f'{base_folder_name}/watermarked_texts.csv'

    partition_size = 200
    start = 1 + (cfg.partition - 1) * partition_size
    end = 1 + cfg.partition * partition_size

    pipeline = PipeLineBuilder(cfg.generator_args)
    model = pipeline.model
    tokenizer = pipeline.tokenizer
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    generator_kwargs = {
        "max_new_tokens": cfg.generator_args.max_new_tokens,
        "do_sample": cfg.generator_args.do_sample,
        "temperature": cfg.generator_args.temperature,
        "top_p": cfg.generator_args.top_p,
        "top_k": cfg.generator_args.top_k,
        "repetition_penalty": cfg.generator_args.repetition_penalty
    }

    for prompt_num in range(start,end):

        prompt, id = get_prompt_and_id_dev(cfg.prompt_file, prompt_num)
            
        log.info(f"Prompt: {prompt}")
        log.info(f"Prompt ID: {id}")

        try:
            for _ in range(1):
                start = time.time()

                watermarked_text = generate(prompt, cfg, model, tokenizer, **generator_kwargs)

                delta = time.time() - start
                
                log.info(f"Time taken: {delta}")

                stats = [{'id': id, 'text': watermarked_text, 'watermarking_scheme': "none", 'model': cfg.generator_args.model_name_or_path, 'time': delta}]
                save_to_csv(stats, watermarked_text_file_path, rewrite=False)
        except Exception as e:
            log.info(f"Exception with Prompt {prompt_num}.")
            log.info(f"Exception: {e}")

if __name__ == "__main__":
    test()
