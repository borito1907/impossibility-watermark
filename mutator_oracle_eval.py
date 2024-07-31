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
    RelativeOracle
)
log = logging.getLogger(__name__)

# from langchain.globals import set_debug; set_debug(True)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def eval(cfg):

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.cuda_visible_devices)
    os.environ["WORLD_SIZE"] = str(len(str(cfg.cuda_visible_devices).split(",")))

    # Tucking import here because 'import torch' prior to setting CUDA_VISIBLE_DEVICES causes an error
    # https://discuss.pytorch.org/t/runtimeerror-device-0-device-num-gpus-internal-assert-failed/178118/6
    from model_builders.pipeline import PipeLineBuilder
    # from watermark import Watermarker
    # from oracle import (
    #     RankOracle,
    #     JointOracle,
    #     RelativeOracle,
    #     SoloOracle
    # )
    from mutators.document import DocumentMutator
    from mutators.sentence import SentenceMutator
    from mutators.span import SpanMutator
    from mutators.word import WordMutator
    # Set number of mutation steps to analyze
    mutation_steps = 100
    log.info(f"Setting number of mutation steps to {mutation_steps}...")

    # Load test data
    # NOTE: we will reuse the outputs from the quality oracle tests
    log.info("Loading tests...")
    tests_df = pd.read_csv("./results/mutated_text.csv")
    log.info(tests_df)
    oracle_config = {"type": "guidance", "class": RelativeOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-70B-Instruct-q8_0.gguf", "explain": False}
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
    oracles = []
    # for t, c in templates:
    #     cfg.oracle_args.template = t
    #     oracles.append(c(cfg=cfg.oracle_args, pipeline=pipeline))

    # Init mutators

    # Construct eval loop
    results = []
    for index, row in tqdm(tests_df.iterrows(), desc='Tests'): 
        out_dict = {}
        out_dict.update(row)
        text = row["mutated_text"]
        # Evaluate Mutation Quality
        choose = "response_a"
        try:
            evals = oracle.is_quality_preserved(row["prompt"], row[choose], text)
            is_quality_preserved = evals["quality_preserved"]
        except Exception as e:
            print("-"*20)
            print(f"ERROR ERRO ERROR: {e}")
            print("-"*20)
            with open("mutator_testing.errors", "a") as f:
                f.write(e)
            is_quality_preserved = "Unknown"
            evals = {}

        out_dict.update({
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

        # log.info(f"Test {index}: {out_dict}")
        results.append(out_dict)

        # Incremental saving over time...
        log.info("Saving results to csv...")
        df = pd.DataFrame(results)
        df.to_csv("./results/mutator_eval.csv", index=False)

    tests_df = pd.read_csv("./results/mutator_eval.csv")
    data = {}
    output = []
    for index, row in tqdm(tests_df.iterrows(), desc='Tests'):
        if row["mutator"] not in data:
            data[row["mutator"]] = {row["mutation_step"]: [0,0]}
        if row["mutation_step"] not in data[row["mutator"]]:
            data[row["mutator"]][row["mutation_step"]] = [0,0]
        data[row["mutator"]][row["mutation_step"]][1] += 1
        if row["quality_preserved"]:
            data[row["mutator"]][row["mutation_step"]][0] += 1
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
    
    fig, ax = plt.subplots(len(data))
    for i, mut in enumerate(data.keys()):
        ax[i].plot(data[mut].keys(), [i[0]/i[1] for i in data[mut].values()])
        ax[i].set_title(f"Percentage vs steps for {mut}")
    fig.tight_layout(pad=5.0)
    plt.show()
    plt.savefig("mutator.png")
    
if __name__ == "__main__":
    eval()
