# RUN: CUDA_VISIBLE_DEVICES=7 python -m oracles.eval_oracles

import logging
import traceback
from oracles import (
    SoloOracle, RankOracle, JointOracle, RelativeOracle, BinaryOracle,
    PrometheusAbsoluteOracle, PrometheusRelativeOracle, 
    ArmoRMOracle, InternLMOracle, OffsetBiasOracle
)
from guidance import models 
from oracles.base import ResponseQuality

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def WQE_row_to_label(row):
    if row["winner_model_a"]:
        return ResponseQuality.A_BETTER
    if row["winner_model_b"]:
        return ResponseQuality.B_BETTER
    if row["winner_tie"]:
        return ResponseQuality.TIE

def IMP_row_to_label(row):
    if "original" in row["selected"]:
        return ResponseQuality.A_BETTER
    if "mutated" in row["selected"]:
        return ResponseQuality.B_BETTER
    if "tie" in row["selected"]:
        return ResponseQuality.TIE

def run_eval():

    import pandas as pd
    import time

    oracles = [
        # RewardBench
        {"type": "rewardbench", "class": OffsetBiasOracle, "llm_path": "NCSOFT/Llama-3-OffsetBias-RM-8B", "explain": False},
        {"type": "rewardbench", "class": InternLMOracle, "llm_path": "internlm/internlm2-20b-reward", "explain": False},
        {"type": "rewardbench", "class": ArmoRMOracle, "llm_path": "RLHFlow/ArmoRM-Llama3-8B-v0.1", "explain": False},

        # Prometheus 
        # {"type": "prometheus", "class": PrometheusAbsoluteOracle, "llm_path": "gpt-4-turbo", "explain": True},
        # {"type": "prometheus", "class": PrometheusRelativeOracle, "llm_path": "gpt-4-turbo", "explain": True}, 
        # {"type": "prometheus", "class": PrometheusAbsoluteOracle, "llm_path": "gpt-4o", "explain": True},
        # {"type": "prometheus", "class": PrometheusRelativeOracle, "llm_path": "gpt-4o", "explain": True}, 
        # {"type": "prometheus", "class": PrometheusAbsoluteOracle, "llm_path": "prometheus-eval/prometheus-8x7b-v2.0", "explain": True},
        # {"type": "prometheus", "class": PrometheusRelativeOracle, "llm_path": "prometheus-eval/prometheus-8x7b-v2.0", "explain": True},

        # Llama-3.1-8B + explain=False
        {"type": "guidance", "class": SoloOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-8B-Instruct-q8_0.gguf", "explain": False},
        {"type": "guidance", "class": RankOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-8B-Instruct-q8_0.gguf", "explain": False},
        {"type": "guidance", "class": JointOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-8B-Instruct-q8_0.gguf", "explain": False},
        {"type": "guidance", "class": RelativeOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-8B-Instruct-q8_0.gguf", "explain": False},
        {"type": "guidance", "class": BinaryOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-8B-Instruct-q8_0.gguf", "explain": False},

        # Llama-3.1-8B + explain=True
        {"type": "guidance", "class": SoloOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-8B-Instruct-q8_0.gguf", "explain": True},
        {"type": "guidance", "class": RankOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-8B-Instruct-q8_0.gguf", "explain": True},
        {"type": "guidance", "class": JointOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-8B-Instruct-q8_0.gguf", "explain": True},
        {"type": "guidance", "class": RelativeOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-8B-Instruct-q8_0.gguf", "explain": True},
        {"type": "guidance", "class": BinaryOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-8B-Instruct-q8_0.gguf", "explain": True},

        # Llama-3.1-70B + explain=False
        {"type": "guidance", "class": SoloOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-q8_0.gguf", "explain": False},
        {"type": "guidance", "class": RankOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-q8_0.gguf", "explain": False},
        {"type": "guidance", "class": JointOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-q8_0.gguf", "explain": False},
        {"type": "guidance", "class": RelativeOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-q8_0.gguf", "explain": False},
        {"type": "guidance", "class": BinaryOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-q8_0.gguf", "explain": False},

        # Llama-3.1-70B + explain=True
        {"type": "guidance", "class": SoloOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-q8_0.gguf", "explain": True},
        {"type": "guidance", "class": RankOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-q8_0.gguf", "explain": True},
        {"type": "guidance", "class": JointOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-q8_0.gguf", "explain": True},
        {"type": "guidance", "class": RelativeOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-q8_0.gguf", "explain": True},
        {"type": "guidance", "class": BinaryOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-q8_0.gguf", "explain": True},
        
    ]

    tests_df = pd.read_csv("./data/IMP/dev.csv")
    print(tests_df)

    save_path = "./oracles/results/IMP_oracle_eval.csv"

    # Check if the save file already exists and load it
    try:
        existing_annotations = pd.read_csv(save_path)
    except FileNotFoundError:
        existing_annotations = pd.DataFrame()
		
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
                oracle = oracle_config["class"](
                    model_id=oracle_config["llm_path"]
                )
            else:
                oracle = oracle_config["class"](
                    model_id=oracle_config["llm_path"],
                    download_dir="/data2/.shared_models",
                    num_gpus=4
                )
            judge_name = oracle_config["llm_path"]
        elif "rewardbench" in oracle_config["type"]:
            oracle = oracle_config["class"]()
            judge_name = oracle_config["llm_path"]
        
        # Iterate over oracle tests
        for benchmark_id, row in tests_df.iterrows():

            out = {**row}
            out.update({"benchmark_id": benchmark_id})

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
                print(f"Skipping already annotated row: id={row['id']}, oracle_class={oracle_class}, judge_name={judge_name}, explain={explain}")
                continue

            try:
                start = time.time()
                test_eval = oracle.test(
                    instruction=row["prompt"], 
                    response_A=row["original_response"], 
                    response_B=row["mutated_response"],
                    label=IMP_row_to_label(row)
                )
                time_taken = time.time() - start
                out.update({
                    "oracle_type": oracle_config['type'],
                    "oracle_class": oracle_class,
                    "judge_name": judge_name,
                    "explain": explain,
                    "time_taken": time_taken,
                    **test_eval
                })
                log.info(out)
                results.append(out)
            except Exception:
                error = traceback.format_exc()     
                print(f"Error on row: id={row['id']}, oracle_class={oracle_class}, judge_name={judge_name}, explain={explain}")
                out.update({
                    "oracle_type": oracle_config['type'],
                    "oracle_class": oracle_class,
                    "judge_name": judge_name,
                    "explain": explain,
                    "time_taken": None,
                    "error": error
                })
                log.info(out)
                results.append(out)

            # Append new results to existing annotations and save to CSV
            df = pd.DataFrame(results)
            keys = ["id", "oracle_class", "judge_name", "explain", "watermark", "mutator", "step"]
            combined_df = pd.concat([existing_annotations, df]).drop_duplicates(subset=keys)
            combined_df.to_csv(save_path, index=False)

        del oracle

if __name__ == "__main__":
    run_eval()
