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

print(f"Total number of traces: {len(traces)}")

samples = []
sampled_group_ids = set() 
sampled_prompts = set()  # Track unique prompts
sampled_combos = set()

# Loop through the traces
for trace in traces:
    o, w, m, s = os.path.basename(trace).split("_")[:4]
    s = s.replace("n-steps=", "")
    prompt = (o, w, m)  # Create a unique tuple of oracle, watermarker, mutator to represent a prompt
    print(f"Processing: {o}, {w}, {m}, {s}")

    # Ensure the prompt is unique
    if prompt in sampled_prompts:
        continue  # Skip if this prompt has already been sampled

    df = assign_unique_group_ids(pd.read_csv(trace, nrows=10000))

    df['watermark_score'] = df['watermark_score'].replace(-1, np.nan)

    df["oracle"] = o
    df["watermarker"] = w
    df["mutator"] = m
    df["n_steps"] = s

    tries = 0

    for _ in range(2):

        if "group_id" in df.columns:
            available_group_ids = df["group_id"].unique()
            # Filter out group_ids that have already been sampled
            unsampled_group_ids = [gid for gid in available_group_ids if gid not in sampled_group_ids]

            # If no unsampled group_ids left, move to next trace
            if not unsampled_group_ids:
                print(f"No unsampled group_ids left in trace {trace}")
                continue

            # Stay within the current trace until we find a valid group_id
            found_valid_group = False
            while unsampled_group_ids and not found_valid_group:
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
                    continue  # Try the next group_id if no valid step_0_row is found

                # Safely extract the row with the lowest watermark score where quality_preserved == True
                quality_preserved_sample = df_sample[df_sample['quality_preserved'] == True]


                # If no valid rows with quality_preserved == True and watermark_score, try the next group_id
                if quality_preserved_sample.empty or not quality_preserved_sample['watermark_score'].notna().any():
                    print(f"No valid rows with quality_preserved == True for group_id {group_id}")
                    tries += 1
                    if tries > 10:
                        break
                    continue  # Try the next group_id

                # Get the row with the lowest watermark score
                try:
                    min_watermark_row = quality_preserved_sample.loc[quality_preserved_sample['watermark_score'].idxmin()]
                except ValueError:
                    print(f"No valid watermark score found for quality_preserved == True in group_id {group_id}")
                    continue  # Try the next group_id

                # Ensure that the min_watermark_row is not the same as the step_0_row
                if step_0_row.equals(min_watermark_row):
                    print(f"Skipped group_id {group_id} because step_0_row and min_watermark_row are the same.")
                    continue  # Try the next group_id

                # If we have valid rows, append them and mark this group_id as valid
                samples.append(step_0_row)
                samples.append(min_watermark_row)
                found_valid_group = True  # Move to the next trace now that we found a valid group_id

        
        unsampled_group_ids.remove(group_id)  # Remove it from the list of unsampled ids
        sampled_group_ids.add(group_id)  # Mark this group_id as sampled
        sampled_prompts.add(prompt)

        # Break once we have 10 unique prompts (which should result in 20 rows total)
        if len(sampled_prompts) >= 20 and len(samples) == 40:
            break

# Ensure we have exactly 20 rows (10 unique group_ids, 10 unique prompts)
df_combined = pd.DataFrame(samples)

# Remove any columns that start with "Unnamed"
df_combined = df_combined.loc[:, ~df_combined.columns.str.contains('^Unnamed')]

# Print the number of unique group_ids and total number of rows
print(f"Unique group_ids: {len(df_combined['group_id'].unique())}")
print(f"Unique prompts: {len(sampled_prompts)}")
print(f"Total rows: {len(df_combined)}")

# Ensure that the DataFrame contains exactly 20 rows and 10 unique prompts
if len(df_combined) == 20 and len(sampled_prompts) == 10:
    print("DataFrame contains the correct number of rows and unique prompts.")
else:
    print("DataFrame does not contain the expected number of rows or unique prompts.")

# Save to CSV
df_combined.to_csv("./human_study/data/attack_traces.csv", index=False)
