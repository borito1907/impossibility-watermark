# RUN: python -m oracles.training.torchtune.convert

import glob
import torch
from tqdm import tqdm
from safetensors.torch import save_file

epoch_id = 4
model_file_path = f"/data2/.shared_models/models--meta-llama--Meta-Llama-3.1-70B-Instruct-WQE-0.1"

pt_to_merge = glob.glob(f"{model_file_path}/hf_model_00*_{epoch_id}.pt")
state_dicts = [torch.load(p) for p in tqdm(pt_to_merge)]
merged_state_dicts = {k: v for d in state_dicts for k, v in d.items()}
# torch.save(merged_state_dicts, f"{model_file_path}/model.bin")
save_file(merged_state_dicts, f"{model_file_path}/model.safetensors")