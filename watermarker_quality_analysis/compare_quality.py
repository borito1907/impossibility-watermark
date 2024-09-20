# RUN: CUDA_VISIBLE_DEVICES=6,7 python -m watermarker_quality_analysis.compare_quality

import traceback
import pandas as pd
from watermarker_factory import get_default_watermarker
from extractors import (
    FluencyMetric, 
    GrammarMetric, 
    QualityMetric,
    MATTRDiversity,
    UniqueBigramsDiversity,
)

import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
logging.getLogger('optimum.gptq.quantizer').setLevel(logging.WARNING)

def main():

    def check_watermark(df, return_mean=True):
        watermarks_detected, watermark_scores = [], []
        for benchmark_id, row in df.iterrows():
            watermark_detected, watermark_score = watermarker_fn.detect(row['text'])
            watermarks_detected.append(watermark_detected)
            watermark_scores.append(watermark_score)
        watermarks_detected = np.array(watermarks_detected)
        watermark_scores = np.array(watermark_scores)
        if return_mean:
            watermarks_detected = watermarks_detected.mean()
            watermark_scores = watermark_scores.mean()
        return {
            "watermark_detected": watermarks_detected,
            "watermark_scores": watermark_scores,
        }

    def get_watermark_scores(unwatermarked_data, watermarked_data):
        mean_unwatermark_score = check_watermark(unwatermarked_data)
        mean_watermark_score = check_watermark(watermarked_data)
        return {
            "mean_unwatermark_score": mean_unwatermark_score,
            "mean_watermark_score": mean_watermark_score,
        }

    def get_perplexities(unwatermarked_data, watermarked_data):
        mean_unwatermarked_perplexity = perplexity.evaluate(unwatermarked_data['text'])
        mean_watermarked_perplexity = perplexity.evaluate(watermarked_data['text'])
        return {
            "mean_unwatermarked_perplexity": mean_unwatermarked_perplexity,
            "mean_watermarked_perplexity": mean_watermarked_perplexity,
        }
        
    def get_grammaticality(unwatermarked_data, watermarked_data):
        mean_unwatermarked_grammar_errors = grammar.evaluate(unwatermarked_data['text'])
        mean_watermarked_grammar_errors = grammar.evaluate(watermarked_data['text'])
        return {
            "mean_unwatermarked_grammar_errors": mean_unwatermarked_grammar_errors,
            "mean_watermarked_grammar_errors": mean_watermarked_grammar_errors,
        }

    def get_quality(unwatermarked_data, watermarked_data):
        mean_unwatermarked_quality = quality.evaluate(unwatermarked_data['prompt'], unwatermarked_data['text'])
        mean_watermarked_quality = quality.evaluate(watermarked_data['prompt'], watermarked_data['text'])
        return {
            "mean_unwatermarked_quality": mean_unwatermarked_quality,
            "mean_watermarked_quality": mean_watermarked_quality,
        }

    def get_diversities(unwatermarked_data, watermarked_data):
        unwatermarked_mattr = mattr.evaluate(unwatermarked_data['text'])
        watermarked_mattr = mattr.evaluate(watermarked_data['text'])
        unwatermarked_bigram = bigram.evaluate(unwatermarked_data['text'])
        watermarked_bigram = bigram.evaluate(watermarked_data['text'])
        return {
            "unwatermarked_mattr": unwatermarked_mattr,
            "watermarked_mattr": watermarked_mattr,
            "unwatermarked_bigram": unwatermarked_bigram,
            "watermarked_bigram": watermarked_bigram,
        }

    watermarkers = [
        "umd",
        "semstamp", 
        "adaptive",
    ]

    # Load default, non-watermarked dataset for comparison
    unwatermarked_data_path = "./data/WQE/dev.csv"
    unwatermarked_data = pd.read_csv(unwatermarked_data_path)

    # Initialize metric extractors
    fluency = FluencyMetric()
    grammar = GrammarMetric()
    quality = QualityMetric()        
    mattr   = MATTRDiversity(),
    bigram  = UniqueBigramsDiversity()

    results = []
    for watermarker in watermarkers:

        # Load data for that particular watermarker
        watermarked_data_path = f"./data/WQE_{watermarker}/dev.csv"
        watermarked_data = pd.read_csv(watermarked_data_path)

        # Initialize watermark detector
        watermarker_fn = get_default_watermarker(watermarker)

        # Collect evaluation statistics
        results.append({
            "watermarker": watermarker, 
            "unwatermarked_data_path": unwatermarked_data_path,
            "watermarked_data_path": watermarked_data_path,
            **get_watermark_scores(unwatermarked_data, watermarked_data),
            **get_perplexities(unwatermarked_data, watermarked_data)
            **get_grammaticality(unwatermarked_data, watermarked_data)
            **get_quality(unwatermarked_data, watermarked_data)
            **get_diversities(unwatermarked_data, watermarked_data)
        })

    df = pd.DataFrame(results)
    df.to_csv("./watermarker_quality_analysis/watermarker_quality_analysis_results.csv", index=False)
    print(df)

if __name__ == "__main__":
    main()
