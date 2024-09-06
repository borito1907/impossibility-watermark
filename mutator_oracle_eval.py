import os
import time
import pandas as pd
import hydra
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from guidance import models 
from extractors import FluencyMetric, GrammarMetric
from oracles import (
    DiffOracle
)
log = logging.getLogger(__name__)

# from langchain.globals import set_debug; set_debug(True)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def eval(cfg):

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.cuda_visible_devices)
    os.environ["WORLD_SIZE"] = str(len(str(cfg.cuda_visible_devices).split(",")))

    # Tucking import here because 'import torch' prior to setting CUDA_VISIBLE_DEVICES causes an error
    # https://discuss.pytorch.org/t/runtimeerror-device-0-device-num-gpus-internal-assert-failed/178118/6

    tests_df = pd.read_csv("./data/mutated_eval.csv")
    log.info(tests_df)
    oracle_config = {"type": "guidance", "class": DiffOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-IMP-0.1-q8_0.gguf", "explain": False}
    llm = models.LlamaCpp(
                model=oracle_config["llm_path"],
                echo=False,
                n_gpu_layers=-1,
                n_ctx=2048
            )
    oracle = oracle_config["class"](llm, explain=oracle_config["explain"])
    judge_name = oracle_config["llm_path"].split("/data2/.shared_models/llama.cpp_models/")[-1].replace("/ggml-model", "")

    fluency = FluencyMetric()
    grammar = GrammarMetric()
    # Construct eval loop

    for index, row in tqdm(tests_df.iterrows(), desc='Tests'): 
        out_dict = {}
        out_dict.update(row)
        text = row["mutated_text"]
        # Evaluate Mutation Quality
        choose = "response_a"
        start_time = time.time()
        try:
            evals = oracle.is_quality_preserved(row["prompt"], row[choose], text)
            is_quality_preserved = evals["quality_preserved"]
        except Exception as e:
            print("-"*20)
            print(f"ERROR ERRO ERROR: {e}")
            print("-"*20)
            with open("mutator_testing.errors", "a") as f:
                f.write(f"{e}\n")
            is_quality_preserved = "Unknown"
            evals = {}
        oracle_time = time.time() - start_time
        out_dict.update({
            "oracle_time": oracle_time,
            "oracle": judge_name,
            "quality_preserved": is_quality_preserved,
            **evals
        })

        # Evaluate Fluency (via Perplexity)
        fluency_score = fluency.evaluate([text])

        # Evaluate Grammar Errors
        count_grammar_errors = grammar.evaluate([text])

        out_dict.update({
            "fluency_score": fluency_score,
            "count_grammar_errors": count_grammar_errors,
        })


        # Incremental saving over time...
        log.info("Saving results to csv...")
        df = pd.DataFrame([out_dict])
        df.to_csv("./results/mutator_eval.csv", header=not os.path.exists("./results/mutator_eval.csv"), mode="a", index=False)

    tests_df = pd.read_csv("./results/mutator_eval.csv")
    data = {}
    output = []
    for index, row in tqdm(tests_df.iterrows(), desc='Tests'):
        if row["mutator"] not in data:
            data[row["mutator"]] = {row["mutation_step"]: {"total": 0, "quality": 0, "fluency": 0, "grammar": 0}}
        if row["mutation_step"] not in data[row["mutator"]]:
            data[row["mutator"]][row["mutation_step"]] = {"total": 0, "quality": 0, "fluency": 0, "grammar": 0}
        data[row["mutator"]][row["mutation_step"]]["total"] += 1
        data[row["mutator"]][row["mutation_step"]]["fluency"] += row["fluency_score"]
        data[row["mutator"]][row["mutation_step"]]["fluency"] += row["count_grammar_errors"]
        if row["quality_preserved"]:
            data[row["mutator"]][row["mutation_step"]]["quality"] += 1
    print(data)
    for i in data:
        for j in data[i]:
            temp = {}
            temp["mutator"] = i
            temp["mutation step"] = j
            temp["percent preserved"] = data[i][j][0]/ data[i][j][1]
            output.append(temp)
    df = pd.DataFrame(output)
    df.to_csv("./results/mutator_percent.csv")
    fig, ax = plt.subplots(3, len(data.keys()), figsize=(12, 15))
    for i, mut in enumerate(data.keys()):
        ax[0][i].plot(data[mut].keys(), [i[1]/i[0] for i in data[mut].values()], label="quality preserved")
        ax[0][i].set_title(f"Percentage vs steps for {mut}")
        ax[0][i].set_xlabel("Number of mutation steps")
        ax[0][i].set_ylabel("Percentage of \n quality preserved")
        # ax[0].set_ylim([0,1.2])
        ax[1][i].plot(data[mut].keys(), [i[2]/i[0] for i in data[mut].values()], label="quality preserved")
        ax[1][i].set_title(f"Percentage vs steps for {mut}")
        ax[1][i].set_xlabel("Number of mutation steps")
        ax[1][i].set_ylabel("Average \n fluency score")
        # ax[1].set_ylim([0,1.2])
        ax[2][i].plot(data[mut].keys(), [i[3]/i[0] for i in data[mut].values()], label="quality preserved")
        ax[2][i].set_title(f"Percentage vs steps for {mut}")
        ax[2][i].set_xlabel("Number of mutation steps")
        ax[2][i].set_ylabel("Average \n Grammar errors")
        ax[2][i].set_ylim([0,1.2])
    # fig.tight_layout(pad=5.0)
    plt.show()
    plt.savefig("mutator.png")
    
if __name__ == "__main__":
    eval()
