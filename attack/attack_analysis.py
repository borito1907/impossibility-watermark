# RUN: CUDA_VISIBLE_DEVICES=4 python -m attack.attack_metrics

import pandas as pd
import matplotlib.pyplot as plt

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


if __name__ == "__main__":

    import os
    import glob

    traces = glob.glob("./attack_traces/*attack_results.csv")

    orig = pd.read_csv("./data/WQE/dev.csv").to_dict()
    domains = {}
    for i in range(136):
        domains[orig['prompt'][i]] = {"high": orig["high_domain"][i], "mid": orig["mid_domain"][i], "low": orig["low_domain"][i], "entropy": orig["entropy_level"][i]}
    watermark_thresholds ={
        "UMDWatermarker": [0.0, 0.25, 0.5, 1.0, 2.0, 3.0],
        "SemStampWatermarker": [0.0, 0.25, 0.5, 1.0, 2.0, 3.0],
        "AdaptiveWatermarker": [50.0, 60.0, 70.0, 80.0, 90.0], 
    }
    testing = {
        "UMDWatermarker": 1.0,
        "SemStampWatermarker": 1.0,
        "AdaptiveWatermarker": 70.0, 
    }
    results = {}
    for trace in traces:

        o, w, m, compare_against_original = os.path.basename(trace).split("_")[:4]
        compare_against_original = "True" in compare_against_original
        
        df = assign_unique_group_ids(pd.read_csv(trace))

        threshold_results = {}
        for t in watermark_thresholds[w]:
            prompts = get_successully_attacked_rows(df,t)['prompt']
            one = False
            hd = {}
            md = {}
            ld = {}
            el = {}
            for prompt in prompts:
                if prompt not in domains:
                    # print(prompt)
                    continue
                one = True
                hd[domains[prompt]['high']] = hd.get(domains[prompt]['high'], 0) + 1
                md[domains[prompt]['mid']] = md.get(domains[prompt]['mid'], 0) + 1
                ld[domains[prompt]['low']] = ld.get(domains[prompt]['low'], 0) + 1
                el[domains[prompt]['entropy']] = el.get(domains[prompt]['entropy'], 0) + 1

            if one:
                hd = {i:hd[i] for i in hd if pd.notna(i)}
                md = {i:md[i] for i in md if pd.notna(i)}
                ld = {i:ld[i] for i in ld if pd.notna(i)}
                el = {i:el[i] for i in el if pd.notna(i)}
                for i in ["low", "mid", "high"]:
                    if i not in el:
                        el[i] = 0
                threshold_results[t] = {"hd": hd, "md": md, "ld": ld, "el": el}
            if t == testing[w]:
                results[f"{w}_{m}_{compare_against_original}"] = {i: el[i]/sum(el.values()) for i in el}
    fig, ax = plt.subplots(3, figsize = (10, 10))
    ax[0].barh(list(results.keys()), [results[i]['high'] for i in results.keys()])
    ax[0].set_title("High entropy frequnecy")
    ax[1].barh(list(results.keys()), [results[i]['low'] for i in results.keys()])
    ax[1].set_title("Low entropy frequnecy")
    ax[2].barh(list(results.keys()), [results[i]['mid'] for i in results.keys()])
    ax[2].set_title("Mid entropy frequnecy")
    fig.tight_layout()
    plt.savefig("./attack/graphs/entropies.png")



        # fig, ax = plt.subplots(len(threshold_results.keys()), 4, figsize=(100, 25))
        # for i, thresh in enumerate(threshold_results.keys()):

        #     ax[i][0].barh(*zip(*threshold_results[thresh]['hd'].items()))
        #     ax[i][0].set_title(f"Count of high domain for threshold {thresh}")

        #     ax[i][1].barh(*zip(*threshold_results[thresh]['md'].items()))
        #     ax[i][1].set_title(f"Count of mid domain for threshold {thresh}")

        #     ax[i][2].barh(*zip(*threshold_results[thresh]['ld'].items()))
        #     ax[i][2].set_title(f"Count of low domain for threshold {thresh}")

        #     ax[i][3].barh(*zip(*threshold_results[thresh]['el'].items()))
        #     ax[i][3].set_title(f"Count of entropy levels for threshold {thresh}")
        # name = f"{os.path.basename(trace).rstrip('.csv')}.png"

        # plt.savefig(f"./attack/graphs/{name}")
        # plt.clf()