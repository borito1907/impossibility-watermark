import pandas as pd
from tqdm import tqdm
if __name__ == "__main__":

    import os
    import glob

    # output_csv = "./attack_traces/total.csv"

    traces1 = glob.glob("./attack_traces/*attack_results_annotated*.csv")
    orig = pd.read_csv("./data/WQE/dev.csv").to_dict()
    combined_df = None
    domains = {}
    for i in range(134):
        domains[orig['prompt'][i]] = {"high": orig["high_domain"][i], "mid": orig["mid_domain"][i], "low": orig["low_domain"][i], "entropy": orig["entropy_level"][i], "id": orig["id"][i]}
    for trace in tqdm(traces1):
        print(trace)
        o, w, m, compare_against_original = os.path.basename(trace).split("_")[:4]
        if "Sem" in trace and "annotated.csv" not in trace:
            continue
        compare_against_original = "True" in compare_against_original
        try:
            cur = pd.read_csv(trace)
        except:
            continue
        t = 0
        for idx, row in tqdm(cur.iterrows()):
            step = row['step_num']
            if idx == 0:
                step = -1
                t = row['prompt']
            if row['prompt'] != t:
                t = row['prompt']
                step = -1
            # print(row)
            if not pd.notna(row['watermark_score']) or row['watermark_score'] == None or int(row['watermark_score']) == -1:
                # break
                continue
            result = {
                "mutator": m,
                "id": domains[row['prompt']]['id'],
                "step_num": step,
                "domain": domains[row['prompt']]['high'],
                "entropy": domains[row['prompt']]['entropy'],
                "watermark_detected": int(row['watermark_detected']),
                "watermark_score": row['watermark_score'],
                # "group_id": row['group_id']
                }
            output_csv = f"./attack_traces/{w}_total.csv"
            result_df = pd.DataFrame([result])
            result_df.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)