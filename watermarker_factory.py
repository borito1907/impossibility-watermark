# This is here to fix circular imports.

from watermarkers import UMDWatermarker, UnigramWatermarker, EXPWatermarker, SemStampWatermarker, AdaptiveWatermarker

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
