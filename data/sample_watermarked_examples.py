import pandas as pd 

prompts = pd.read_csv("./data/WQE/dev.csv")[["id", "prompt"]]
samples = pd.read_csv("./data/WQE/dev_watermarked.csv")

samples = pd.merge(samples, prompts, how="inner", on="id")

n_samples = 100
cols = ["id", "prompt", "text", "zscore", "watermarking_scheme", "model"]

# filter for zscore > 3
samples = samples[samples["zscore"] >=3].sample(n_samples)[cols]

samples.to_csv("./data/wqe_watermark_samples.csv", index=False)
