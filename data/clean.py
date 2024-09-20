# RUN: python -m data.clean

import pandas as pd
from langdetect import detect, LangDetectException

def filter_non_english(df, text_column):
    """
    Filters out non-English rows from a pandas DataFrame based on the detected language of a given text column.

    :param df: Input pandas DataFrame
    :param text_column: The name of the column containing text data
    :return: DataFrame with only English text
    """
    def is_english(text):
        try:
            return detect(text) == 'en'
        except LangDetectException:
            # If language detection fails (e.g., for empty or invalid text), return False
            return False

    # Apply language detection and filter only rows where language is English
    df['is_english'] = df[text_column].apply(is_english)
    english_df = df[df['is_english']].drop(columns=['is_english'])  # Drop the helper column

    return english_df

for split in ["dev", "test"]:

    original_df = pd.read_csv(f"./data/WQE/{split}.csv")

    print(split, len(original_df))

    # Filter out non-english texts
    original_df = filter_non_english(original_df, 'prompt')

    print(split, len(original_df))

    original_df.to_csv(f"./data/WQE/{split}.csv", encoding="utf-8", index=False)

    for wm in ["adaptive", "semstamp", "umd", "unwatermarked"]:

        wm_df = pd.read_csv(f"./data/WQE_{wm}/{split}.csv", index_col=False)
        
        # Filter wm_df to only include rows that exist in original_df based on 'id'
        wm_df = wm_df[wm_df['id'].isin(original_df['id'])]

        # Merge the 'prompt' column from original_df into wm_df on 'id'
        if "prompt" not in wm_df.columns:
            wm_df = wm_df.merge(original_df[['id', 'prompt']], on='id', how='left')     

        # Trim newlines and spaces in the 'text' field of wm_df
        wm_df['text'] = wm_df['text'].str.strip()

        # Remove 'Unnamed' columns
        wm_df = wm_df.loc[:, ~wm_df.columns.str.contains('^Unnamed')]

        print(split, wm, len(original_df))

        # Filter out non-english texts
        wm_df = filter_non_english(wm_df, 'text')

        print(split, wm, len(original_df))

        # Reordering columns
        key_cols = ['id', 'prompt', 'text']
        columns_order = key_cols + [c for c in wm_df.columns if c not in key_cols]
        wm_df = wm_df[columns_order]

        wm_df.to_csv(f"./data/WQE_{wm}/{split}.csv", encoding="utf-8", index=False)