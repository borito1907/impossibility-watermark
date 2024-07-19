# RUN: CUDA_VISIBLE_DEVICES=4,5,6,7 python -m oracles.eval_oracles

import logging
from guidance import models 
from oracles import (
    SoloOracle, RankOracle, JointOracle, RelativeOracle,
    PrometheusAbsoluteOracle, PrometheusRelativeOracle
)
from oracles.base import ResponseQuality

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('optimum.gptq.quantizer').setLevel(logging.WARNING)

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
        # don't explain answers before giving them - takes less time and better for efficiency
        {"type": "guidance", "class": SoloOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct/ggml-model-q8_0.gguf", "explain": False},
        {"type": "guidance", "class": SoloOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-70B-Instruct/ggml-model-q8_0.gguf", "explain": False},
        {"type": "guidance", "class": RankOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct/ggml-model-q8_0.gguf", "explain": False},
        {"type": "guidance", "class": RankOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-70B-Instruct/ggml-model-q8_0.gguf", "explain": False},
        {"type": "guidance", "class": JointOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct/ggml-model-q8_0.gguf", "explain": False},
        {"type": "guidance", "class": JointOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-70B-Instruct/ggml-model-q8_0.gguf", "explain": False},
        {"type": "guidance", "class": RelativeOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct/ggml-model-q8_0.gguf", "explain": False},
        {"type": "guidance", "class": RelativeOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-70B-Instruct/ggml-model-q8_0.gguf", "explain": False},
        # explain answers before giving them - will take longer but should enable higher accuracy
        {"type": "guidance", "class": SoloOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct/ggml-model-q8_0.gguf", "explain": True},
        {"type": "guidance", "class": SoloOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-70B-Instruct/ggml-model-q8_0.gguf", "explain": True},
        {"type": "guidance", "class": RankOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct/ggml-model-q8_0.gguf", "explain": True},
        {"type": "guidance", "class": RankOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-70B-Instruct/ggml-model-q8_0.gguf", "explain": True},
        {"type": "guidance", "class": JointOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct/ggml-model-q8_0.gguf", "explain": True},
        {"type": "guidance", "class": JointOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-70B-Instruct/ggml-model-q8_0.gguf", "explain": True},
        {"type": "guidance", "class": RelativeOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct/ggml-model-q8_0.gguf", "explain": True},
        {"type": "guidance", "class": RelativeOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-70B-Instruct/ggml-model-q8_0.gguf", "explain": True},
        # prometheus models always provide an explanation
        # {"type": "prometheus", "class": PrometheusAbsoluteOracle, "llm_path": "prometheus-eval/prometheus-8x7b-v2.0", "explain": True},
        # {"type": "prometheus", "class": PrometheusAbsoluteOracle, "llm_path": "gpt-4o", "explain": True},
        # {"type": "prometheus", "class": PrometheusRelativeOracle, "llm_path": "prometheus-eval/prometheus-8x7b-v2.0", "explain": True},
        # {"type": "prometheus", "class": PrometheusRelativeOracle, "llm_path": "gpt-4o", "explain": True}, # not working well at the moment: https://github.com/prometheus-eval/prometheus-eval/issues/46
    ]

    tests_df = pd.read_csv("./tests/quality_oracle/lmsys_tiny.csv")
		
    eval_results = []
    for oracle_config in oracles:     

        # Initialize Oracle
        if "guidance" in oracle_config["type"]:
            llm = models.LlamaCpp(
                model=oracle_config["llm_path"],
                echo=False,
                n_gpu_layers=-1,
                n_ctx=2048
            )
            oracle = oracle_config["class"](llm, explain=oracle_config["explain"])
            model_name = oracle_config["llm_path"].split("/data2/.shared_models/llama.cpp_models/")[-1].replace("/ggml-model", "")
        elif "prometheus" in oracle_config["type"]:
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
            model_name = oracle_config["llm_path"]
        
        # Iterate over oracle tests
        for benchmark_id, row in tests_df.iterrows():

            start = time.time()
            test_eval = oracle.test(
                instruction=row["prompt"], 
                response_A=row["response_a"], 
                response_B=row["response_b"],
                label=lmsys_row_to_label(row)
            )
            time_taken = time.time() - start
            test_eval.update({
                "benchmark_id": benchmark_id, 
                "oracle_type": oracle_config['type'],
                "oracle_class": oracle_config['class'].__name__,
                "judge_name": model_name,
                "explain": oracle_config['explain'],
                "time_taken": time_taken,
            })
            log.info(test_eval)
            eval_results.append(test_eval)

            df = pd.DataFrame(eval_results)
            df.to_csv(f"./oracles/results/oracle_eval.csv")    
        

if __name__ == "__main__":
    run_eval()
