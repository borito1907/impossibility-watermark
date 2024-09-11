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

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.cuda_visible_devices)
    os.environ["WORLD_SIZE"] = str(len(str(cfg.cuda_visible_devices).split(",")))

    # Tucking import here because 'import torch' prior to setting CUDA_VISIBLE_DEVICES causes an error
    # https://discuss.pytorch.org/t/runtimeerror-device-0-device-num-gpus-internal-assert-failed/178118/6
    from model_builders.pipeline import PipeLineBuilder
    # from watermark import Watermarker
    from oracles import (
        RankOracle,
        JointOracle,
        RelativeOracle,
        SoloOracle,
        DiffOracle
    )
    from mutators import (
        DocumentMutator, SentenceMutator, WordMutator,  
        SpanMutator, Span2Mutator, Span3Mutator, Span4Mutator 
    )
    
    # Set number of mutation steps to analyze
    mutation_steps = 100
    log.info(f"Setting number of mutation steps to {mutation_steps}...")

    # Load test data
    # NOTE: we will reuse the outputs from the quality oracle tests
    log.info("Loading tests...")
    tests_df = pd.read_csv("./data/WQE_adaptive/dev.csv")
    log.info(tests_df)
    oracle_config = {"type": "guidance", "class": DiffOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-IMP-DiffOracle-0.1-q8_0.gguf", "explain": False}
    oracle = oracle_config["class"](explain=oracle_config["explain"])
    judge_name = oracle_config["llm_path"].split("/data2/.shared_models/llama.cpp_models/")[-1].replace(".ggml", "")

    fluency = FluencyMetric()
    grammar = GrammarMetric()
    oracles = []
    # for t, c in templates:
    #     cfg.oracle_args.template = t
    #     oracles.append(c(cfg=cfg.oracle_args, pipeline=pipeline))

    # Init mutators
    log.info(f"Initializing mutators...")
    # doc_mutator = DocumentMutator()
    # sent_mutator = SentenceMutator(cfg.oracle_args)
    # span_mutator = SpanMutator()
    # word_mutator = WordMutator()

    mutators = [SpanMutator, Span2Mutator, Span3Mutator, Span4Mutator]

    # Construct eval loop
    results = []
    for index, row in tqdm(tests_df.iterrows(), desc='Tests'): 
        for mutator_class in tqdm(mutators, desc='Mutators'):
            mutator = mutator_class()
            # if mutator_name == "doc":
            #     mutator = DocumentMutator()
            # elif mutator_name == "sent":
            #     mutator = SentenceMutator(cfg.oracle_args) 
            # elif mutator_name == "span":
            #     mutator = SpanMutator()
            # elif mutator_name == "word":
            #     mutator = WordMutator()
            choose = "response_a"
            # if row["winner_model_a"] == "1":
            #     choose = "response_a"
            # else:
            #     choose = "response_b"
            text = row[choose]

            for mutation_step in tqdm(range(mutation_steps)):

                # Initialize output_dict
                out_dict = {}
                out_dict.update(row)

                # Mutate text
                start = time.time()
                try:
                    text = mutator.mutate(text)
                except Exception as e:
                    print("-"*20)
                    print(f"ERROR ERRO ERROR: {e}")
                    print("-"*20)
                    continue

                mutation_time = time.time() - start

                out_dict.update({
                    "mutator": mutator.__class__.__name__,
                    "mutated_text": text,
                    "mutation_step": mutation_step+1,
                    "mutation_time": mutation_time,
                })
                
                # Evaluate Mutation Quality
                try:
                    evals = oracle.is_quality_preserved(row["prompt"], row[choose], text)
                    is_quality_preserved = evals["quality_preserved"]
                except Exception as e:
                    print("-"*20)
                    print(f"ERROR ERRO ERROR: {e}")
                    print("-"*20)
                    with open("mutator_testing.errors", "a") as f:
                        f.write(str(e))
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
            del mutator

    tests_df = pd.read_csv("./results/mutator_eval.csv")
    data = {}
    output = []
    for index, row in tqdm(tests_df.iterrows(), desc='Tests'):
        if row["mutator"] not in data:
            data[row["mutator"]] = {row["mutation_step"]: [0,0,0,0]}
        if row["mutation_step"] not in data[row["mutator"]]:
            data[row["mutator"]][row["mutation_step"]] = [0,0,0,0]
        data[row["mutator"]][row["mutation_step"]][0] += 1
        if row["quality_preserved"]:
            data[row["mutator"]][row["mutation_step"]][1] += 1
        data[row["mutator"]][row["mutation_step"]][2] += row["fluency_score"]
        data[row["mutator"]][row["mutation_step"]][3] += row["count_grammar_errors"]
    print(data)
    for i in data:
        for j in data[i]:
            temp = {}
            temp["mutator"] = i
            temp["mutation_step"] = j
            temp["percent_preserved"] = data[i][j][1]/ data[i][j][0]
            temp["fluency_score"] = data[i][j][2]/ data[i][j][0]
            temp["count_grammar_errors"] = data[i][j][3]/ data[i][j][0]
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

    plt.show()
    plt.savefig("mutator3.png")
    
if __name__ == "__main__":
    eval()
