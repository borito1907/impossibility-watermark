# RUN: CUDA_VISIBLE_DEVICES=2,7 python -m watermarker_quality_analysis.compare_quality

import os
import traceback
from tqdm import tqdm
import numpy as np
import pandas as pd
from watermarker_factory import get_default_watermarker
from extractors import (
    FluencyMetric, 
    GrammarMetric, 
    QualityMetric,
)

import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
logging.getLogger('optimum.gptq.quantizer').setLevel(logging.WARNING)

def main():

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
        if "watermark_scores" not in df.columns:
            df = check_watermark(df)

        # Compute perplexity (fluency)
        if "perplexity" not in df.columns:
            df = fluency.evaluate_dataframe(df, text_column=text_column, new_column='perplexity')

        # Compute grammaticality
        if "grammar_errors" not in df.columns:
            df = grammar.evaluate_dataframe(df, text_column=text_column, new_column='grammar_errors')

        # Compute quality
        if "skywork27B_quality" not in df.columns:
            df = quality.evaluate_dataframe(df, prompt_column=prompt_column, text_column=text_column, new_column='skywork27B_quality')

        return df

    watermarkers = ["umd_new", "adaptive_neo", "unigram"]

    # Load unwatermarked dataset
    unwatermarked_data_path = "./data/WQE_unwatermarked/dev.csv"
    unwatermarked_data = pd.read_csv(unwatermarked_data_path)

    # Initialize metric extractors
    fluency = FluencyMetric(device="cpu")
    grammar = GrammarMetric()
    quality = QualityMetric()

    for watermarker in watermarkers:
        print(watermarker)

        # Load watermarked data
        watermarked_data_path = f"./data/WQE_{watermarker}/dev.csv"
        watermarked_data = pd.read_csv(watermarked_data_path)

        # Initialize watermark detector
        watermarker_fn = get_default_watermarker(watermarker)

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
        if "watermark_scores" in df.columns:
            out["watermark_scores_mean"] = df["watermark_scores"].mean()
            out["watermark_scores_std"] = df["watermark_scores"].std()

        # Compute perplexity (fluency)
        if "perplexity" in df.columns:
            out["perplexity_mean"] = df["perplexity"].mean()
            out["perplexity_std"] = df["perplexity"].std()

        # Compute grammaticality
        if "grammar_errors" in df.columns:
            out["grammar_errors_mean"] = df["grammar_errors"].mean()
            out["grammar_errors_std"] = df["grammar_errors"].std()

        # Compute quality
        if "skywork27B_quality" in df.columns:
            out["skywork27B_quality_mean"] = df["skywork27B_quality"].mean()
            out["skywork27B_quality_std"] = df["skywork27B_quality"].std()

        results.append(out)
    
    results_df = pd.DataFrame(results)
    print(results_df)

    results_df.to_csv("./watermarker_quality_analysis/watermarker_quality_analysis_results_v2.csv")


if __name__ == "__main__":
    # main()
    summarize()
 