import pandas as pd
import glob
import random
import os
import numpy as np

def assign_unique_group_ids(df):
    df['new_group'] = (df['step_num'] == 0).astype(int)
    df['group_id'] = df['new_group'].cumsum()
    return df

# Collect all traces that match the pattern
traces = glob.glob("./attack_traces/*attack_results_annotated*.csv")

# Shuffle the list of traces
random.shuffle(traces)

print(len(traces))

samples = []
sampled_group_ids = set() 
sampled_combos = set()

# Loop through the traces
for trace in traces:

    o, w, m, s = os.path.basename(trace).split("_")[:4]
    s = s.replace("n-steps=", "")
    print(o, w, m, s)

    if (w, m) in sampled_combos:
        continue
    else:
        sampled_combos.add((w, m))

    df = assign_unique_group_ids(pd.read_csv(trace))

    df['watermark_score'] = df['watermark_score'].replace(-1, np.nan)

    df["oracle"] = o
    df["watermarker"] = w
    df["mutator"] = m
    df["n_steps"] = s

    if "group_id" in df.columns:
        available_group_ids = df["group_id"].unique()
        # Filter out group_ids that have already been sampled
        unsampled_group_ids = [gid for gid in available_group_ids if gid not in sampled_group_ids]
        
        # Only proceed if there are unsampled group_ids
        if unsampled_group_ids:
            # Sample a new group_id
            group_id = random.choice(unsampled_group_ids)
            
            # Filter the DataFrame for the sampled group_id
            df_sample = df[df["group_id"] == group_id]
            
            # Safely attempt to extract the first row with step_num == -1 or 0
            step_0_row = None
            try:
                if not df_sample[df_sample['step_num'] == -1].empty:
                    step_0_row = df_sample[df_sample['step_num'] == -1].iloc[0]
                elif not df_sample[df_sample['step_num'] == 0].empty:
                    step_0_row = df_sample[df_sample['step_num'] == 0].iloc[0]
            except IndexError:
                print(f"No step_num == -1 or 0 found in group_id {group_id}")

            # Safely extract the row with the lowest watermark score where quality_preserved == True
            quality_preserved_sample = df_sample[df_sample['quality_preserved'] == True]

            # If no valid rows with quality_preserved == True and watermark_score, skip this group
            if quality_preserved_sample.empty or not quality_preserved_sample['watermark_score'].notna().any():
                print(f"No valid rows with quality_preserved == True for group_id {group_id}")
                continue  # Skip to the next iteration
            
            # Get the row with the lowest watermark score
            try:
                min_watermark_row = quality_preserved_sample.loc[quality_preserved_sample['watermark_score'].idxmin()]
            except ValueError:
                print(f"No valid watermark score found for quality_preserved == True in group_id {group_id}")
                continue  # Skip to the next iteration
            
            # Append the rows if they exist
            if step_0_row is not None:
                samples.append(step_0_row)
            if min_watermark_row is not None and (min_watermark_row.name != step_0_row.name if step_0_row is not None else True):
                samples.append(min_watermark_row)

            sampled_group_ids.add(group_id)  # Mark this group_id as sampled
    
    # Break once we have 10 group_id samples
    if len(sampled_group_ids) >= 10:
        break

# Convert list of samples to a DataFrame
df_combined = pd.DataFrame(samples)

# Remove any columns that start with "Unnamed"
df_combined = df_combined.loc[:, ~df_combined.columns.str.contains('^Unnamed')]

# Print the number of unique group_ids
print(len(df_combined))
print(len(df_combined['group_id'].unique()))

# Save to CSV
df_combined.to_csv("./human_study/data/attack_traces.csv", index=False)
