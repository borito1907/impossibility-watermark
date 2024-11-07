# RUN: CUDA_VISIBLE_DEVICES=2,7 python -m watermarker_quality_analysis.compare_quality

import os
import traceback
from tqdm import tqdm
import numpy as np
import pandas as pd
from watermarker_factory import get_watermarker
from extractors import (
    FluencyMetric, 
    GrammarMetric, 
    QualityMetric,
    InternLMQualityMetric
)
import hydra
from omegaconf import OmegaConf
import yaml

import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
logging.getLogger('optimum.gptq.quantizer').setLevel(logging.WARNING)

def get_watermarker_config(cfg, config_path):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    with open(config_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    # Update the config only for existing keys
    for key in yaml_data:
        # log.info(f"key: {key}")
        cfg_dict['watermark_args'][key] = yaml_data[key]

    log.info(f"cfg_dict: {cfg_dict}")
    cfg = OmegaConf.create(cfg_dict)
    return cfg

@hydra.main(version_base=None, config_path="../conf", config_name="gen_conf")
def main(cfg):

    def check_watermark(df, return_mean=True):
        if 'watermark_detected' in df.columns and 'watermark_scores' in df.columns:
            return df  # Return the DataFrame with existing data

        watermarks_detected, watermark_scores = [], []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Running {check_watermark.__name__}"):
            try:
                watermark_detected, watermark_score = watermarker_fn.detect(row['text'])
                watermarks_detected.append(watermark_detected)
                watermark_scores.append(watermark_score)
            except Exception:
                print(traceback.format_exc())
                watermarks_detected.append(np.nan)
                watermark_scores.append(np.nan)
        
        df['watermark_detected'] = watermarks_detected
        df['watermark_scores'] = watermark_scores
        return df  # Return the updated DataFrame

    def process_dataframe(df, text_column='text', prompt_column='prompt'):
        # Process each metric and update the DataFrame with new columns
        
        # Compute watermark detection
        # if "watermark_scores" not in df.columns:
        #     df = check_watermark(df)

        # Compute perplexity (fluency)
        if "perplexity" not in df.columns:
            df = fluency.evaluate_dataframe(df, text_column=text_column, new_column='perplexity')

        # Compute grammaticality
        if "grammar_errors" not in df.columns:
            df = grammar.evaluate_dataframe(df, text_column=text_column, new_column='grammar_errors')

        if "early_grammar_errors" not in df.columns:
            df = grammar.evaluate_dataframe(df, text_column=text_column, new_column='early_grammar_errors', early=True)

        # # Compute quality
        # if "skywork27B_quality" not in df.columns:
        #     df = quality.evaluate_dataframe(df, prompt_column=prompt_column, text_column=text_column, new_column='skywork27B_quality')

        if "internlm_quality" not in df.columns:
            df = intern_quality.evaluate_dataframe(df, prompt_column=prompt_column, text_column=text_column, new_column='internlm_quality')

        return df

    # watermarkers = ["adaptive_neo", "adaptive_neo_delta0.5", "adaptive_neo_delta1.0", "WQE_adaptive_neo_delta0.5_nosecret", "WQE_adaptive_neo_delta1.5_alpha3", "WQE_adaptive_neo_delta1.5_alpha4"]
    # watermarkers = ["adaptive_neo_delta0.5_nosecret", "adaptive_neo_delta1.5_alpha3", "adaptive_neo_delta1.5_alpha4"]
    # watermarkers = ["umd_new", "semstamp", "adaptive_neo", "adaptive_neo_delta0.5", "adaptive_neo_delta1.0"]
    watermarkers = ["adaptive_delta0.25_alpha4.0_nosecret"]
    # watermarkers = ["adaptive_delta0.50_alpha3.0_nosecret"]
    # watermarkers = ["adaptive_delta1.00_alpha3.0_nosecret"]
    # watermarkers = ["adaptive_delta0.25_alpha4"]

    # Load unwatermarked dataset
    unwatermarked_data_path = "./data/WQE_unwatermarked/dev.csv"
    unwatermarked_data = pd.read_csv(unwatermarked_data_path)

    # Initialize metric extractors
    fluency = FluencyMetric(device="cpu")
    grammar = GrammarMetric()
    # quality = QualityMetric()
    intern_quality = InternLMQualityMetric()

    for watermarker in watermarkers:
        print(watermarker)

        # Load watermarked data
        watermarked_data_path = f"./data/WQE_{watermarker}/dev.csv"
        watermarked_cfg_path = f"./data/WQE_{watermarker}/conf.yaml"
        watermarked_data = pd.read_csv(watermarked_data_path)
        watermarker_cfg = get_watermarker_config(cfg, watermarked_cfg_path)

        watermarker_fn = get_watermarker(watermarker_cfg)

        # Process the unwatermarked and watermarked datasets
        print("Processing unwatermarked data...")
        unwatermarked_data = process_dataframe(unwatermarked_data)

        print("Processing watermarked data...")
        watermarked_data = process_dataframe(watermarked_data)

        # Save the updated CSVs
        unwatermarked_data.to_csv(unwatermarked_data_path, index=False)
        watermarked_data.to_csv(watermarked_data_path, index=False)
        
def summarize():
    import glob

    results = []
    for path in glob.glob("./data/WQE*/dev.csv"):
        df = pd.read_csv(path)
        out = {"path": path}
        # Compute watermark detection
        # if "watermark_scores" in df.columns:
        #     out["watermark_scores_mean"] = df["watermark_scores"].mean()
        #     out["watermark_scores_std"] = df["watermark_scores"].std()

        if "zscore" in df.columns:
            out["watermark_scores_mean"] = df["zscore"].mean()
            out["watermark_scores_std"] = df["zscore"].std()

        # Compute perplexity (fluency)
        if "perplexity" in df.columns:
            out["perplexity_mean"] = df["perplexity"].mean()
            out["perplexity_std"] = df["perplexity"].std()

        # Compute grammaticality
        if "grammar_errors" in df.columns:
            out["grammar_errors_mean"] = df["grammar_errors"].mean()
            out["grammar_errors_std"] = df["grammar_errors"].std()

        if "early_grammar_errors" in df.columns:
            out["early_grammar_errors_mean"] = df["early_grammar_errors"].mean()
            out["early_grammar_errors_std"] = df["early_grammar_errors"].std()

        # Compute quality
        if "skywork27B_quality" in df.columns:
            out["skywork27B_quality_mean"] = df["skywork27B_quality"].mean()
            out["skywork27B_quality_std"] = df["skywork27B_quality"].std()

        # Compute quality
        if "internlm_quality" in df.columns:
            out["internlm_quality_mean"] = df["internlm_quality"].mean()
            out["internlm_quality_std"] = df["internlm_quality"].std()

        results.append(out)
    
    results_df = pd.DataFrame(results)
    print(results_df)

    results_df.to_csv("./watermarker_quality_analysis/watermarker_quality_analysis_results_v16.csv")


if __name__ == "__main__":
    main()
    # summarize()
 