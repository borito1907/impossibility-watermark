# This is here to fix circular imports.

from watermarkers import UMDWatermarker, UnigramWatermarker, EXPWatermarker, SemStampWatermarker, AdaptiveWatermarker
from omegaconf import OmegaConf
from hydra import initialize, compose

def get_watermarker(cfg, **kwargs):
    if cfg.watermark_args.name == "umd":
        return UMDWatermarker(cfg, **kwargs)
    elif cfg.watermark_args.name == "unigram":
        return UnigramWatermarker(cfg, **kwargs)
    elif cfg.watermark_args.name == "exp":
        return EXPWatermarker(cfg, **kwargs)
    elif "semstamp" in cfg.watermark_args.name:
        return SemStampWatermarker(cfg, **kwargs)
    elif cfg.watermark_args.name == "adaptive":
        return AdaptiveWatermarker(cfg, **kwargs)
    else:
        raise NotImplementedError(f"Watermarker with name {cfg.watermark_args.name}.")


umd_dict = {}
umd_dict['watermark_args'] = {}
umd_dict['watermark_args']['name'] = "umd"
umd_dict['watermark_args']['gamma'] = 0.25
umd_dict['watermark_args']['delta'] = 2.0
umd_dict['watermark_args']['seeding_scheme'] = "selfhash"
umd_dict['watermark_args']['ignore_repeated_ngrams'] = True
umd_dict['watermark_args']['normalizers'] = []
umd_dict['watermark_args']['z_threshold'] = 0.5
umd_dict['watermark_args']['device'] = 'cuda'
umd_dict['watermark_args']['only_detect'] = True
umd_cfg = OmegaConf.create(umd_dict)

semstamp_dict = {}
semstamp_dict['watermark_args'] = {}
semstamp_dict['watermark_args']['name'] = "semstamp_lsh"
semstamp_dict['watermark_args']['embedder'] = {}
semstamp_dict['watermark_args']['delta'] = 0.01
semstamp_dict['watermark_args']['sp_mode'] = "lsh"
semstamp_dict['watermark_args']['sp_dim'] = 3
semstamp_dict['watermark_args']['lmbd'] = 0.25
semstamp_dict['watermark_args']['max_new_tokens'] = 255
semstamp_dict['watermark_args']['min_new_tokens'] = 245
semstamp_dict['watermark_args']['max_trials'] = 50
semstamp_dict['watermark_args']['critical_max_trials'] = 75
semstamp_dict['watermark_args']['cc_path'] = None
semstamp_dict['watermark_args']['train_data'] = None
semstamp_dict['watermark_args']['device'] = "auto"
semstamp_dict['watermark_args']['len_prompt'] = 32
semstamp_dict['watermark_args']['z_threshold'] = 0.5
semstamp_dict['watermark_args']['use_fine_tuned'] = False
semstamp_dict['watermark_args']['only_detect'] = True
semstamp_cfg = OmegaConf.create(semstamp_dict)

adaptive_dict = {}
adaptive_dict['watermark_args'] = {}
adaptive_dict['generator_args'] = {}
adaptive_dict['watermark_args']['name'] = "adaptive"
adaptive_dict['watermark_args']['measure_model_name'] = "gpt2-large"
adaptive_dict['watermark_args']['embedding_model_name'] = "sentence-transformers/all-mpnet-base-v2"
adaptive_dict['watermark_args']['delta'] = 1.5
adaptive_dict['watermark_args']['delta_0'] = 1.0
adaptive_dict['watermark_args']['alpha'] = 2.0
adaptive_dict['watermark_args']['top_k'] = 50
adaptive_dict['watermark_args']['top_p'] = 0.9
adaptive_dict['watermark_args']['max_new_tokens'] = 786 # 285
adaptive_dict['watermark_args']['min_new_tokens'] = 128 # 215
adaptive_dict['watermark_args']['only_detect'] = True
adaptive_dict['watermark_args']['device'] = 'auto'
adaptive_dict['watermark_args']['detection_threshold'] = 95.0
adaptive_dict['generator_args']['model_name_or_path'] = "Llama-3.1"
adaptive_dict['watermark_args']['secret_string'] = 'The quick brown fox jumps over the lazy dog'
adaptive_dict['watermark_args']['measure_threshold'] = 50
adaptive_cfg = OmegaConf.create(adaptive_dict)

def get_default_watermarker(watermarker_name):
    if watermarker_name == "umd":
        return UMDWatermarker(umd_cfg)
    elif watermarker_name == "semstamp":
        return SemStampWatermarker(semstamp_cfg)
    elif watermarker_name == "adaptive":
        return AdaptiveWatermarker(adaptive_cfg)
    else:
        raise NotImplementedError(f"Watermarker with name {watermarker_name}.")