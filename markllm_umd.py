import torch
from markllm.watermark.auto_watermark import AutoWatermark
from markllm.utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import hydra
import logging
from model_builders.pipeline import PipeLineBuilder

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="gen_conf")
def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = PipeLineBuilder(cfg.generator_args)
    
    model = pipeline.model
    tokenizer = pipeline.tokenizer
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    # Transformers config
    transformers_config = TransformersConfig(model=model.to(device),
                                            tokenizer=tokenizer,
                                            vocab_size=tokenizer.vocab_size,
                                            device=device,
                                            max_new_tokens=cfg.generator_args.max_new_tokens,
                                            min_length=cfg.generator_args.min_new_tokens,
                                            do_sample=cfg.generator_args.do_sample,
                                            no_repeat_ngram_size=cfg.generator_args.no_repeat_ngram_size)
    
    # Load watermark algorithm
    myWatermark = AutoWatermark.load('KGW', 
                                    algorithm_config='config/KGW.json',
                                    transformers_config=transformers_config)

    # Prompt
    prompt = "Explain the daily life of a citizen in Ancient Greece."

    # Generate and detect
    watermarked_text = myWatermark.generate_watermarked_text(prompt)
    detect_result = myWatermark.detect_watermark(watermarked_text)
    log.info(f"Watermarked Text: {watermarked_text}")
    log.info(f"Detect Result: {detect_result}")
    unwatermarked_text = myWatermark.generate_unwatermarked_text(prompt)
    detect_result = myWatermark.detect_watermark(unwatermarked_text)
    log.info("-" * 100)
    log.info(f"Unwatermarked Text: {watermarked_text}")
    log.info(f"Detect Result: {detect_result}")

if __name__ == "__main__":
    main()