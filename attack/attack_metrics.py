# RUN: CUDA_VISIBLE_DEVICES=4 python -m attack.attack_metrics

import pandas as pd
from extractors import FluencyMetric, GrammarMetric, QualityMetric

def assign_unique_group_ids(df):
    df['new_group'] = (df['step_num'] == 0).astype(int)
    df['group_id'] = df['new_group'].cumsum()
    return df

def get_support(df):
    return (df['step_num'] == 0).sum()

def get_max_step_count(df):
    return df['step_num'].max()

def get_successully_attacked_rows(df):
    return df[(df['quality_preserved'] == True) & (df['watermark_detected'] == False)]

def get_mean_step_count_to_break_watermark(df):
    successful_df = get_successully_attacked_rows(df)
    if successful_df.empty:
        return 0
    return successful_df["step_num"].mean()

def get_attack_success_rate(df):
    success_count = len(get_successully_attacked_rows(df))
    divisor = get_support(df)
    if divisor == 0:
        return 0
    success_rate = success_count / divisor
    return success_rate

def get_mean_mutation_time(df):
    return df["mutator_time"].mean()

def get_mean_oracle_time(df):
    return df["oracle_time"].mean()

def get_mean_change_in_z_scores(df):
    quality_preserved_df = df[df['quality_preserved'] == True]
    quality_preserved_df = quality_preserved_df.sort_values(by='step_num')
    quality_preserved_df['watermark_score_change'] = quality_preserved_df['watermark_score'].diff()
    average_change = quality_preserved_df['watermark_score_change'].mean()
    return average_change

def get_mean_total_time_for_successful_attacks(df):
    successful_df = get_successully_attacked_rows(df)
    successful_group_ids = successful_df['group_id'].unique()
    successful_groups = df[df['group_id'].isin(successful_group_ids)]
    total_time_per_group = successful_groups.groupby('group_id')['total_time'].sum()
    mean_total_time = total_time_per_group.mean()
    return mean_total_time

def get_original_and_final_text_comparisons(df):
    original_text_df = df[df['step_num'] == 0][['group_id', 'prompt', 'current_text']]
    final_text_df = df[(df['quality_preserved'] == True) & (df['watermark_detected'] == False)][['group_id', 'mutated_text']]
    comparison_df = pd.merge(original_text_df, final_text_df, on='group_id', how='inner')
    comparison_df.rename(columns={'current_text': 'original_text', 'mutated_text': 'final_mutated_text'}, inplace=True)
    return comparison_df

def get_fluencies_on_successful_attacks(df):
    fluency = FluencyMetric()
    comparison_df = get_original_and_final_text_comparisons(df)
    mean_original_fluency = fluency.evaluate(comparison_df['original_text'])
    mean_attacked_fluency = fluency.evaluate(comparison_df['final_mutated_text'])
    return {
        "mean_original_fluency": mean_original_fluency,
        "mean_attacked_fluency": mean_attacked_fluency,
    }
    
def get_grammaticality_on_successful_attacks(df):
    grammar = GrammarMetric()
    comparison_df = get_original_and_final_text_comparisons(df)
    mean_original_grammar_errors = grammar.evaluate(comparison_df['original_text'])
    mean_attacked_grammar_errors = grammar.evaluate(comparison_df['final_mutated_text'])
    return {
        "mean_original_grammar_errors": mean_original_grammar_errors,
        "mean_attacked_grammar_errors": mean_attacked_grammar_errors,
    }

def get_quality_on_successful_attacks(df):
    quality = QualityMetric()
    comparison_df = get_original_and_final_text_comparisons(df)
    mean_original_quality = quality.evaluate(comparison_df['prompt'], comparison_df['original_text'])
    mean_original_quality = quality.evaluate(comparison_df['prompt'], comparison_df['final_mutated_text'])
    return {
        "mean_original_quality": mean_original_quality,
        "mean_attacked_quality": mean_original_quality,
    }
   

if __name__ == "__main__":

    import os
    import glob

    traces = glob.glob("./attack_traces/*attack_results.csv")

    results = []
    for trace in traces:

        o, w, m, compare_against_origin = os.path.basename(trace).split("_")[:4]
        compare_against_origin = "True" in compare_against_origin

        df = assign_unique_group_ids(pd.read_csv(trace))

        results.append({
            "oracle": o,
            "watermarker": w,
            "mutator": m,
            "compare_against_origin": compare_against_origin,
            "attack_success_rate": get_attack_success_rate(df),
            "mean_steps_to_break_watermark_when_successful": get_mean_step_count_to_break_watermark(df),
            "max_steps": get_max_step_count(df),
            "mean_mutation_time_in_s": get_mean_mutation_time(df),
            "mean_oracle_time_in_s": get_mean_oracle_time(df),
            "mean_total_time_for_successful_attacks_in_s": get_mean_total_time_for_successful_attacks(df),
            "mean_change_in_z_scores": get_mean_change_in_z_scores(df),
            **get_fluencies_on_successful_attacks(df),
            **get_grammaticality_on_successful_attacks(df),
            **get_quality_on_successful_attacks(df),
            "support": get_support(df),
        })

    df = pd.DataFrame(results)
    print(df)