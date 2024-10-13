import pandas as pd
import glob
import os

def assign_unique_group_ids(df):
    df['new_group'] = (df['step_num'] == 0).astype(int)
    df['group_id'] = df['new_group'].cumsum()
    return df

traces = glob.glob("./attack_traces/*attack_results_annotated.csv")

errors = []
for trace in traces:

    print(trace)

    o, w, m, s = os.path.basename(trace).split("_")[:4]
    s = int(s.replace("n-steps=", ""))
    try:
        df = assign_unique_group_ids(pd.read_csv(trace))
        df["mutated_text"] = df["mutated_text"].fillna(df["current_text"])
        df['current_text'] = df['mutated_text'].shift(1)
        df["current_text"] = df["current_text"].fillna(df["mutated_text"])
    except:
        errors.append(trace)

print(errors)
print(len(errors))
