# RUN: CUDA_VISIBLE_DEVICES=2,3,4,5 python -m oracles.eval_oracles

import logging
import traceback
from guidance import models 
from oracles import (
    SoloOracle, RankOracle, JointOracle, RelativeOracle,
    PrometheusAbsoluteOracle, PrometheusRelativeOracle, BinaryOracle
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
        # {"type": "guidance", "class": SoloOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct-q8.gguf", "explain": False},
        # {"type": "guidance", "class": RankOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct-q8.gguf", "explain": False},
        # {"type": "guidance", "class": JointOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct-q8.gguf", "explain": False},
        # {"type": "guidance", "class": RelativeOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct-q8.gguf", "explain": False},
        # {"type": "guidance", "class": SoloOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-70B-Instruct-q8.gguf", "explain": False},
        # {"type": "guidance", "class": RankOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-70B-Instruct-q8.gguf", "explain": False},
        # {"type": "guidance", "class": JointOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-70B-Instruct-q8.gguf", "explain": False},
        # {"type": "guidance", "class": RelativeOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-70B-Instruct-q8.gguf", "explain": False},
        # # Q8 + Explain=True
        # {"type": "guidance", "class": SoloOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct-q8.gguf", "explain": True},
        # {"type": "guidance", "class": RankOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct-q8.gguf", "explain": True},
        # {"type": "guidance", "class": JointOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct-q8.gguf", "explain": True},
        # {"type": "guidance", "class": RelativeOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct-q8.gguf", "explain": True},
        # {"type": "guidance", "class": SoloOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-70B-Instruct-q8.gguf", "explain": True},
        # {"type": "guidance", "class": RankOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-70B-Instruct-q8.gguf", "explain": True},
        # {"type": "guidance", "class": JointOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-70B-Instruct-q8.gguf", "explain": True},
        # {"type": "guidance", "class": RelativeOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-70B-Instruct-q8.gguf", "explain": True},
        # # prometheus models always provide an explanation
        # {"type": "prometheus", "class": PrometheusAbsoluteOracle, "llm_path": "gpt-4-turbo", "explain": True},
        # {"type": "prometheus", "class": PrometheusRelativeOracle, "llm_path": "gpt-4-turbo", "explain": True}, 
        # {"type": "prometheus", "class": PrometheusAbsoluteOracle, "llm_path": "gpt-4o", "explain": True},
        # {"type": "prometheus", "class": PrometheusRelativeOracle, "llm_path": "gpt-4o", "explain": True}, 
        # {"type": "prometheus", "class": PrometheusAbsoluteOracle, "llm_path": "prometheus-eval/prometheus-8x7b-v2.0", "explain": True},
        # {"type": "prometheus", "class": PrometheusRelativeOracle, "llm_path": "prometheus-eval/prometheus-8x7b-v2.0", "explain": True},
        {"type": "guidance", "class": BinaryOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-8B-Instruct-q8_0.gguf", "explain": False},
        {"type": "guidance", "class": BinaryOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-8B-Instruct-q8_0.gguf", "explain": True},
    ]

    tests_df = pd.read_csv("./data/lmsys-150-test-set.csv")

    save_path = "./oracles/results/binary_test.csv"

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
            judge_name = oracle_config["llm_path"].split("/data2/.shared_models/llama.cpp_models/")[-1].replace("/ggml-model", "")
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
        
        # Iterate over oracle tests
        for benchmark_id, row in tests_df.iterrows():

            out = {**row}

            # Skip rows that have already been annotated
            if not existing_annotations.empty and (
                (existing_annotations["id"]           == row["id"])    & 
                (existing_annotations["oracle_class"] == oracle_class) & 
                (existing_annotations["judge_name"]   == judge_name)   & 
                (existing_annotations["explain"]      == explain)
            ).any():
                print(f"Skipping already annotated row: id={row['id']}, oracle_class={oracle_class}, judge_name={judge_name}, explain={explain}")
                continue

            try:
                start = time.time()
                test_eval = oracle.test(
                    instruction=row["prompt"], 
                    response_A=row["response_a"], 
                    response_B=row["response_b"],
                    label=lmsys_row_to_label(row)
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

                # Append new results to existing annotations and save to CSV
                df = pd.DataFrame(results)
                combined_df = pd.concat([existing_annotations, df]).drop_duplicates(subset=["id", "oracle_class", "judge_name", "explain"])
                combined_df.to_csv(save_path, index=False)
            except Exception:
                print(traceback.format_exc())                
                print(f"Error on row: id={row['id']}, oracle_class={oracle_class}, judge_name={judge_name}, explain={explain}")

if __name__ == "__main__":
    run_eval()
