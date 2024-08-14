import pandas as pd 


human_data = pd.read_csv("./human_study/data/human_data.csv")
human_data = human_data[human_data["user_choice"] != "skipped"]

# id,prompt,text,zscore,watermarking_scheme,model,mutated_text,mutator,mutation_step,mutation_time
important_cols = ["id", "zscore", "model", "prompt", "watermark", "mutator", "step"]
base_data = pd.read_csv("./human_study/data/wqe_watermark_samples_mutated.csv", on_bad_lines='skip')[important_cols]

df = pd.merge(human_data, base_data, how="left", on=["prompt", "watermark", "mutator", "step"])

cols = [
    'id', 'prompt', 'model', 'watermark', 'zscore', 
    'original_response', 'mutated_response', 'mutator', 'step', 
    'selected', 'user'
]

print(df[cols].shape)

df.to_csv("./data/IMP/dev.csv", index=False)
