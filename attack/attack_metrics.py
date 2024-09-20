# RUN: CUDA_VISIBLE_DEVICES=5 python -m attack.attack_metrics

import pandas as pd
from extractors import FluencyMetric, GrammarMetric, QualityMetric

# Initialize metric extractors
fluency = FluencyMetric()
grammar = GrammarMetric()
quality = QualityMetric()

def assign_unique_group_ids(df):
    df['new_group'] = (df['step_num'] == 0).astype(int)
    df['group_id'] = df['new_group'].cumsum()
    return df

def get_support(df):
    return (df['step_num'] == 0).sum()

def get_successfully_attacked_support(df, watermark_threshold=0.0):
    df = get_successully_attacked_rows(df, watermark_threshold)
    return len(df['group_id'].unique())

def get_max_step_count(df):
    return df['step_num'].max()

def get_successully_attacked_rows(df, watermark_threshold=0.0):
    successful_df = df[(df['quality_preserved'] == True) & 
                       (df['watermark_score'] < watermark_threshold)]
    successful_df = successful_df.sort_values(by='step_num').groupby('group_id').first().reset_index()
    return successful_df

def get_mean_step_count_to_break_watermark(df, watermark_threshold=0.0):
    successful_df = get_successully_attacked_rows(df, watermark_threshold)
    if successful_df.empty:
        return None
    return successful_df["step_num"].mean()

def get_attack_success_rate(df, watermark_threshold=0.0):
    success_count = len(get_successully_attacked_rows(df, watermark_threshold))
    divisor = get_support(df)
    if divisor == 0:
        return None
    success_rate = success_count / divisor
    return success_rate

def get_mean_mutation_time(df):
    return df["mutator_time"].mean()

def get_mean_oracle_time(df):
    return df["oracle_time"].mean()

def get_mean_change_in_z_scores(df, watermark_threshold=0.0):
    quality_preserved_df = df[df['quality_preserved'] == True].copy()
    quality_preserved_df = quality_preserved_df.sort_values(by='step_num')
    z_score_changes = []
    for group_id, group_df in quality_preserved_df.groupby('group_id'):
        first_success_idx = group_df[group_df['watermark_score'] < watermark_threshold].index.min()
        if pd.notna(first_success_idx):
            group_df = group_df.loc[:first_success_idx]  # Consider only steps before the threshold
        else:
            group_df = group_df  # If no success, consider the whole group
        group_df['watermark_score_change'] = group_df['watermark_score'].diff()
        z_score_changes.extend(group_df['watermark_score_change'].dropna().tolist())
    if z_score_changes:
        mean_change = sum(z_score_changes) / len(z_score_changes)
    else:
        mean_change = None
    return mean_change

def get_mean_total_time_for_successful_attacks(df, watermark_threshold=0.0):
    successful_df = get_successully_attacked_rows(df, watermark_threshold)
    successful_group_ids = successful_df['group_id'].unique()  
    total_times = []
    for group_id in successful_group_ids:
        group_df = df[df['group_id'] == group_id]
        first_success_idx = group_df[group_df['watermark_score'] < watermark_threshold].index.min()
        total_time_before_success = group_df.loc[:first_success_idx, 'total_time'].sum()
        total_times.append(total_time_before_success)
    if total_times:
        mean_total_time = sum(total_times) / len(total_times)
    else:
        mean_total_time = None
    return mean_total_time

def get_original_and_final_text_comparisons(df, watermark_threshold=0.0):
    original_text_df = df[df['step_num'] == 0][['group_id', 'prompt', 'current_text']]
    final_text_df = get_successully_attacked_rows(df, watermark_threshold)[['group_id', 'mutated_text']]
    comparison_df = pd.merge(original_text_df, final_text_df, on='group_id', how='inner')
    comparison_df.rename(columns={'current_text': 'original_text', 'mutated_text': 'final_mutated_text'}, inplace=True)
    return comparison_df

def get_perplexities_on_successful_attacks(df, watermark_threshold=0.0):
    comparison_df = get_original_and_final_text_comparisons(df, watermark_threshold)
    if not comparison_df.empty:
        mean_original_perplexity = fluency.evaluate(comparison_df['original_text'].tolist())
        mean_attacked_perplexity = fluency.evaluate(comparison_df['final_mutated_text'].tolist())
        return {
            f"mean_original_perplexity_@_{watermark_threshold}": mean_original_perplexity,
            f"mean_attacked_perplexity_@_{watermark_threshold}": mean_attacked_perplexity,
        }
    else:
        return {
            f"mean_original_perplexity_@_{watermark_threshold}": None,
            f"mean_attacked_perplexity_@_{watermark_threshold}": None,
        }
    
def get_grammaticality_on_successful_attacks(df, watermark_threshold=0.0):
    comparison_df = get_original_and_final_text_comparisons(df, watermark_threshold)
    if not comparison_df.empty: 
        mean_original_grammar_errors = grammar.evaluate(comparison_df['original_text'].tolist())
        mean_attacked_grammar_errors = grammar.evaluate(comparison_df['final_mutated_text'].tolist())
        return {
            f"mean_original_grammar_errors_@_{watermark_threshold}": mean_original_grammar_errors,
            f"mean_attacked_grammar_errors_@_{watermark_threshold}": mean_attacked_grammar_errors,
        }
    else:
        return {
            f"mean_original_grammar_errors_@_{watermark_threshold}": None,
            f"mean_attacked_grammar_errors_@_{watermark_threshold}": None,
        }

def get_quality_on_successful_attacks(df, watermark_threshold=0.0):
    comparison_df = get_original_and_final_text_comparisons(df, watermark_threshold)
    if not comparison_df.empty:
        mean_original_quality = quality.evaluate(comparison_df['prompt'].tolist(), comparison_df['original_text'].tolist())
        mean_attacked_quality = quality.evaluate(comparison_df['prompt'].tolist(), comparison_df['final_mutated_text'].tolist())
        return {
            f"mean_original_quality_@_{watermark_threshold}": mean_original_quality,
            f"mean_attacked_quality_@_{watermark_threshold}": mean_attacked_quality,
        }
    else:
        return {
            f"mean_original_quality_@_{watermark_threshold}": None,
            f"mean_attacked_quality_@_{watermark_threshold}": None,
        }

if __name__ == "__main__":

    import os
    import glob

    traces = glob.glob("./attack_traces/*attack_results.csv")

    watermark_thresholds ={
        "UMDWatermarker": [0.0, 0.25, 0.5, 1.0, 2.0, 3.0],
        "SemStampWatermarker": [0.0, 0.25, 0.5, 1.0, 2.0, 3.0],
        "AdaptiveWatermarker": [50.0, 60.0, 70.0, 80.0, 90.0], 
    }

    results = []
    for trace in traces:

        o, w, m, compare_against_original = os.path.basename(trace).split("_")[:4]
        compare_against_original = "True" in compare_against_original
        
        df = assign_unique_group_ids(pd.read_csv(trace))

        threshold_results = {}
        for t in watermark_thresholds[w]:
            threshold_results[f"attack_success_@_{t}"] = get_attack_success_rate(df, t)
            threshold_results[f"mean_steps_to_success_@_{t}"] = get_mean_step_count_to_break_watermark(df, t)
            threshold_results[f"mean_time_to_success_@_{t}"] = get_mean_total_time_for_successful_attacks(df, t)
            threshold_results[f"mean_change_in_watermark_score_@_{t}"] = get_mean_change_in_z_scores(df, t)
            threshold_results.update(get_perplexities_on_successful_attacks(df,t))
            threshold_results.update(get_grammaticality_on_successful_attacks(df,t))
            threshold_results.update(get_quality_on_successful_attacks(df,t))
            threshold_results[f"support_@_{t}"] = get_successfully_attacked_support(df, t)

        results.append({
            "oracle": o,
            "watermarker": w,
            "mutator": m,
            "compare_against_original": compare_against_original,
            "mean_mutation_time_in_s": get_mean_mutation_time(df),
            "mean_oracle_time_in_s": get_mean_oracle_time(df),
            **threshold_results,
            "max_steps": get_max_step_count(df),
            "support": get_support(df),
        })

    df = pd.DataFrame(results)
    df.to_csv("./attack_traces/metrics.csv", index=False)
    print(df)