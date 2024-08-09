import pandas as pd 

prompts = pd.read_csv("./data/WQE/dev.csv")[["id", "prompt"]]
umd_samples = pd.read_csv("./human_study/data/dev_umd_watermarked.csv")
semstamp_samples = pd.read_csv("./human_study/data/dev_semstamp_watermarked.csv")

samples = pd.concat([umd_samples, semstamp_samples])
samples = pd.merge(samples, prompts, how="inner", on="id")

n_samples = 100
cols = ["id", "prompt", "text", "zscore", "watermarking_scheme", "model"]

# filter for zscore > 3
samples = samples[samples["zscore"] >=3].sample(n_samples)[cols]

samples.to_csv("./human_study/data/wqe_watermark_samples.csv", index=False)
