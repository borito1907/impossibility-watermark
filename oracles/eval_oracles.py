# RUN: CUDA_VISIBLE_DEVICES=7 python -m oracles.eval_oracles

import os
import logging
import traceback
from guidance import models 
from oracles import (
    SoloOracle, RankOracle, JointOracle, RelativeOracle,
    PrometheusAbsoluteOracle, PrometheusRelativeOracle, 
    BinaryOracle, MutationOracle, Mutation1Oracle, ExampleOracle, DiffOracle,
    ArmoRMOracle, InternLMOracle
)
from oracles.base import ResponseQuality

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def lmsys_row_to_label(row):
    if row["winner_model_a"]:
        return ResponseQuality.A_BETTER
    if row["winner_model_b"]:
        return ResponseQuality.B_BETTER
    if row["winner_tie"]:
        return ResponseQuality.TIE
    
def human_row_to_label(row):
    if row["selected"] == "original":
        return ResponseQuality.A_BETTER
    if row["selected"] == "mutated":
        return ResponseQuality.B_BETTER
    if row["selected"] == "tie":
        return ResponseQuality.TIE

def run_eval():

    import pandas as pd
    import time

    oracles = [
        # # F16 + Explain=False
        # {"type": "guidance", "class": SoloOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct-f16.gguf", "explain": False},
        # {"type": "guidance", "class": RankOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct-f16.gguf", "explain": False},
        # {"type": "guidance", "class": JointOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct-f16.gguf", "explain": False},
        # {"type": "guidance", "class": RelativeOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct-f16.gguf", "explain": False},
        # {"type": "guidance", "class": SoloOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-70B-Instruct-f16.gguf", "explain": False},
        # {"type": "guidance", "class": RankOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-70B-Instruct-f16.gguf", "explain": False},
        # {"type": "guidance", "class": JointOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-70B-Instruct-f16.gguf", "explain": False},
        # {"type": "guidance", "class": RelativeOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-70B-Instruct-f16.gguf", "explain": False},
        # # F16 + Explain=True
        # {"type": "guidance", "class": SoloOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct-f16.gguf", "explain": True},
        # {"type": "guidance", "class": RankOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct-f16.gguf", "explain": True},
        # {"type": "guidance", "class": JointOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct-f16.gguf", "explain": True},
        # {"type": "guidance", "class": RelativeOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct-f16.gguf", "explain": True},
        # {"type": "guidance", "class": SoloOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-70B-Instruct-f16.gguf", "explain": True},
        # {"type": "guidance", "class": RankOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-70B-Instruct-f16.gguf", "explain": True},
        # {"type": "guidance", "class": JointOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-70B-Instruct-f16.gguf", "explain": True},
        # {"type": "guidance", "class": RelativeOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-70B-Instruct-f16.gguf", "explain": True},
        # # Q8 + Explain=False
        # {"type": "guidance", "class": SoloOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct-q8_0.gguf", "explain": False},
        # {"type": "guidance", "class": RankOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct-q8_0.gguf", "explain": False},
        # {"type": "guidance", "class": JointOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct-q8_0.gguf", "explain": False},
        # {"type": "guidance", "class": RelativeOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct-q8_0.gguf", "explain": False},
        # {"type": "guidance", "class": SoloOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-70B-Instruct-q8_0.gguf", "explain": False},
        # {"type": "guidance", "class": RankOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-70B-Instruct-q8_0.gguf", "explain": False},
        # {"type": "guidance", "class": JointOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-70B-Instruct-q8_0.gguf", "explain": False},
        # {"type": "guidance", "class": RelativeOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-70B-Instruct-q8_0.gguf", "explain": False},
        # {"type": "guidance", "class": SoloOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-WQE-0.1-q8_0.gguf", "explain": False},
        # {"type": "guidance", "class": RankOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-WQE-0.1-q8_0.gguf", "explain": False},
        # {"type": "guidance", "class": JointOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-WQE-0.1-q8_0.gguf", "explain": False},
        # {"type": "guidance", "class": RelativeOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-WQE-0.1-q8_0.gguf", "explain": False},
        # # Q8 + Explain=True
        # {"type": "guidance", "class": SoloOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct-q8_0.gguf", "explain": True},
        # {"type": "guidance", "class": RankOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct-q8_0.gguf", "explain": True},
        # {"type": "guidance", "class": JointOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct-q8_0.gguf", "explain": True},
        # {"type": "guidance", "class": RelativeOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct-q8_0.gguf", "explain": True},
        # {"type": "guidance", "class": SoloOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-70B-Instruct-q8_0.gguf", "explain": True},
        # {"type": "guidance", "class": RankOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-70B-Instruct-q8_0.gguf", "explain": True},
        # {"type": "guidance", "class": JointOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-70B-Instruct-q8_0.gguf", "explain": True},
        # {"type": "guidance", "class": RelativeOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-70B-Instruct-q8_0.gguf", "explain": True},
        # {"type": "guidance", "class": SoloOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-WQE-0.1-q8_0.gguf", "explain": True},
        # {"type": "guidance", "class": RankOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-WQE-0.1-q8_0.gguf", "explain": True},
        # {"type": "guidance", "class": JointOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-WQE-0.1-q8_0.gguf", "explain": True},
        # {"type": "guidance", "class": RelativeOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-WQE-0.1-q8_0.gguf", "explain": True},
        # # prometheus models always provide an explanation
        # {"type": "prometheus", "class": PrometheusAbsoluteOracle, "llm_path": "gpt-4-turbo", "explain": True},
        # {"type": "prometheus", "class": PrometheusRelativeOracle, "llm_path": "gpt-4-turbo", "explain": True}, 
        # {"type": "prometheus", "class": PrometheusAbsoluteOracle, "llm_path": "gpt-4o", "explain": True},
        # {"type": "prometheus", "class": PrometheusRelativeOracle, "llm_path": "gpt-4o", "explain": True}, 
        # {"type": "prometheus", "class": PrometheusAbsoluteOracle, "llm_path": "prometheus-eval/prometheus-8x7b-v2.0", "explain": True},
        # {"type": "prometheus", "class": PrometheusRelativeOracle, "llm_path": "prometheus-eval/prometheus-8x7b-v2.0", "explain": True},
        
				# {"type": "guidance", "class": BinaryOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct-q8_0.gguf", "explain": False},
				# {"type": "guidance", "class": MutationOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct-q8_0.gguf", "explain": False},
        # {"type": "guidance", "class": Mutation1Oracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct-q8_0.gguf", "explain": False},
				# {"type": "guidance", "class": SoloOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct-q8_0.gguf", "explain": False},
        # {"type": "guidance", "class": RelativeOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct-q8_0.gguf", "explain": False},
        # {"type": "guidance", "class": ExampleOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct-q8_0.gguf", "explain": False},
        # {"type": "guidance", "class": DiffOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct-q8_0.gguf", "explain": False},
        
        {"type": "guidance", "class": BinaryOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-q8_0.gguf", "explain": False},
				{"type": "guidance", "class": MutationOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-q8_0.gguf", "explain": False},
        {"type": "guidance", "class": Mutation1Oracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-q8_0.gguf", "explain": False},
				{"type": "guidance", "class": SoloOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-q8_0.gguf", "explain": False},
        {"type": "guidance", "class": RelativeOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-q8_0.gguf", "explain": False},
        {"type": "guidance", "class": ExampleOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-q8_0.gguf", "explain": False},
        {"type": "guidance", "class": DiffOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-q8_0.gguf", "explain": False},
        
        # {"type": "guidance", "class": BinaryOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-WQE-0.1-q8_0.gguf", "explain": False},
				# {"type": "guidance", "class": MutationOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-WQE-0.1-q8_0.gguf", "explain": False},
        # {"type": "guidance", "class": Mutation1Oracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-WQE-0.1-q8_0.gguf", "explain": False},
				# {"type": "guidance", "class": SoloOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-WQE-0.1-q8_0.gguf", "explain": False},
        # {"type": "guidance", "class": RelativeOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-WQE-0.1-q8_0.gguf", "explain": False},
        # {"type": "guidance", "class": ExampleOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-WQE-0.1-q8_0.gguf", "explain": False},
        # {"type": "guidance", "class": DiffOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-WQE-0.1-q8_0.gguf", "explain": False},

				# NOTE: These oracles only work with 1 visible GPU, so the script temporarily removes all but one GPUs from the list of visible devices.
        #       Thus, these should be run last, since we don't want those GPUs to be taken when following oracles are run with all given GPUs.
        # 			TLDR: Run these last!! - Connor
				{"type": "rewardbench", "class": ArmoRMOracle, "llm_path": None, "explain": False},
        {"type": "rewardbench", "class": InternLMOracle, "llm_path": None, "explain": False},
    ]
    

    tests_df = pd.read_csv("human_study/data/human_data.csv")
    
    save_path = "./oracles/results/oracle_revamp_eval.csv"

    # Check if the save file already exists and load it
    try:
        existing_annotations = pd.read_csv(save_path)
    except FileNotFoundError:
        existing_annotations = pd.DataFrame()
        

		# In case oracle only works with one device
    ORIGINAL_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]
		
    results = []
    for oracle_config in oracles:    

        oracle_class = oracle_config['class'].__name__ 
        explain = oracle_config['explain']

        # Initialize Oracle
        if "guidance" in oracle_config["type"]:
            llm = models.LlamaCpp(
                model=oracle_config["llm_path"],
                echo=False,
                n_gpu_layers=-1,
                n_ctx=2048
            )
            oracle = oracle_config["class"](llm, explain=oracle_config["explain"])
            judge_name = oracle_config["llm_path"].split("/data2/.shared_models/llama.cpp_models/")[-1].replace(".gguf", "")
            
        elif "prometheus" in oracle_config["type"]:
            print(f"Loading Prometheus with {oracle_config['llm_path']}...")
            if "gpt-" in oracle_config["llm_path"]:
                oracle = PrometheusAbsoluteOracle(
                    model_id=oracle_config["llm_path"]
                )
            else:
                oracle = PrometheusAbsoluteOracle(
                    model_id=oracle_config["llm_path"],
                    download_dir="/data2/.shared_models",
                    num_gpus=4
                )
            judge_name = oracle_config["llm_path"]
            
        elif "rewardbench" in oracle_config['type']:
             if oracle_config['class'] in [ArmoRMOracle, InternLMOracle]:
                 os.environ["CUDA_VISIBLE_DEVICES"] = ORIGINAL_CUDA_VISIBLE_DEVICES.split(',')[0]
                 log.info(f"Setting CUDA_VISIBLE_DEVICES to {ORIGINAL_CUDA_VISIBLE_DEVICES.split(',')[0]}")
                 
             oracle = oracle_config["class"](model=oracle_config['llm_path'], explain=oracle_config["explain"])
             judge_name = oracle_config["llm_path"] #if oracle_config["llm_path"] else "None"
        

        # Iterate over oracle tests
        for benchmark_id, row in tests_df.iterrows():

            out = {**row}

            # Skip rows that have already been annotated
            if not existing_annotations.empty and (
                (existing_annotations["id"]           == row["id"])        & 
                (existing_annotations["watermark"]    == row["watermark"]) & 
                (existing_annotations["mutator"]      == row["mutator"])   & 
                (existing_annotations["step"]         == row["step"])      & 
                (existing_annotations["oracle_class"] == oracle_class)     & 
                (existing_annotations["judge_name"]   == judge_name)       & 
                (existing_annotations["explain"]      == explain)
            ).any():
                print(f"Skipping already annotated row: prompt={row['prompt'][:15]}... step={row['step']}, oracle_class={oracle_class}, judge_name={judge_name}, explain={explain}")
                continue

            try:
                start = time.time()
                test_eval = oracle.test(
                    instruction=row["prompt"], 
                    response_A=row["original_response"], 
                    response_B=row["mutated_response"],
                    label=human_row_to_label(row)
                )
                time_taken = time.time() - start
                out.update({
                    "oracle_type": oracle_config['type'],
                    "oracle_class": oracle_class,
                    "judge_name": judge_name,
                    "explain": explain,
                    "time_taken": time_taken,
                    "label": human_row_to_label(row),
                    **test_eval
                })
                log.info(out)
                results.append(out)

                # Append new results to existing annotations and save to CSV
                df = pd.DataFrame(results)
                combined_df = pd.concat([existing_annotations, df]).drop_duplicates(subset=["id", "oracle_class", "judge_name", "explain", "step", "watermark"])
                combined_df.to_csv(save_path, index=False)
            except Exception:
                print(traceback.format_exc())                
                print(f"Error on row: step={row['step']}, oracle_class={oracle_class}, judge_name={judge_name}, explain={explain}")

        del oracle
        
        if oracle_config['class'] in [ArmoRMOracle, InternLMOracle]:
          os.environ["CUDA_VISIBLE_DEVICES"] = ORIGINAL_CUDA_VISIBLE_DEVICES
          log.info(f"Setting CUDA_VISIBLE_DEVICES to {ORIGINAL_CUDA_VISIBLE_DEVICES}")

if __name__ == "__main__":
    run_eval()
