# RUN: python -m data.clean

import pandas as pd

for split in ["dev", "test"]:

    original_df = pd.read_csv(f"./data/WQE/{split}.csv")

    for wm in ["adaptive", "semstamp", "umd"]:

        wm_df = pd.read_csv(f"./data/WQE_{wm}/{split}.csv", index_col=False)

        # Merge the 'prompt' column from original_df into wm_df on 'id'
        wm_df = wm_df.merge(original_df[['id', 'prompt']], on='id', how='left')     

        # Trim newlines and spaces in the 'text' field of wm_df
        wm_df['text'] = wm_df['text'].str.strip()

        # Remove 'Unnamed' columns
        wm_df = wm_df.loc[:, ~wm_df.columns.str.contains('^Unnamed')]

        # Reordering columns
        key_cols = ['id', 'prompt', 'text']
        columns_order = key_cols + [c for c in wm_df.columns if c not in key_cols]
        wm_df = wm_df[columns_order]

        wm_df.to_csv(f"./data/WQE_{wm}/{split}.csv", encoding="utf-8", index=False)