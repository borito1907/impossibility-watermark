import os
import time
import pandas as pd
import hydra
from tqdm import tqdm
import logging
from oracles.absolute import PrometheusAbsoluteOracle
import matplotlib.pyplot as plt
from watermarker_factory import get_watermarker

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
    watermarker = get_watermarker(cfg, only_detect=False)
    # Load test data
    # NOTE: we will reuse the outputs from the quality oracle tests
    log.info("Loading tests...")
    tests_df = pd.read_csv("./tests/mutator/tests_v2.csv")
    log.info(tests_df)

    # Init shared pipeline for oracles and LLMMutator
    oracles = []
    # for t, c in templates:
    #     cfg.oracle_args.template = t
    #     oracles.append(c(cfg=cfg.oracle_args, pipeline=pipeline))

    # Init mutators
    log.info(f"Initializing mutators...")
    # doc_mutator = DocumentMutator()
    # sent_mutator = SentenceMutator(cfg.oracle_args)
    span_mutator = SpanMutator()
    word_mutator = WordMutator()
    mutators = [word_mutator, span_mutator]

    # Construct eval loop
    results = []
    for index, row in tqdm(tests_df.iterrows(), desc='Tests'): 
        marked_text = watermarker.generate(row["prompt"])
        for mutator in tqdm(mutators, desc='Mutators'):
            text = marked_text

            for mutation_step in range(mutation_steps):

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
                
                try:
                    is_detected, score = watermarker.detect(text)
                except:
                    print("-"*20)
                    print(f"ERROR ERRO ERROR: {e}")
                    print("-"*20)
                    with open("mutator_testing.errors", "a") as f:
                        f.write(e)
                    score = "Unknown"
                    is_detected = "Unknown"

                out_dict.update({
                    "watermarker": cfg.watermark_args.name,
                    "score": score,
                    "is_detected": is_detected
                })


                # log.info(f"Test {index}: {out_dict}")
                results.append(out_dict)

                # Incremental saving over time...
                log.info("Saving results to csv...")
                df = pd.DataFrame(results)
                df.to_csv("./results/mutator_watermark_eval.csv", index=False)

    tests_df = pd.read_csv("./results/mutator_watermark_eval.csv")
    data = {}
    output = []
    for index, row in tqdm(tests_df.iterrows(), desc='Tests'):
        if row["mutator"] not in data:
            data[row["mutator"]] = {row["mutation_step"]: [0,0]}
        if row["mutation_step"] not in data[row["mutator"]]:
            data[row["mutator"]][row["mutation_step"]] = [0,0]
        data[row["mutator"]][row["mutation_step"]][1] += 1
        if row["is_detected"]:
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
    df.to_csv("./results/mutator_watermark_percent.csv")

    fig, ax = plt.subplots(len(data))
    for i, mut in enumerate(data.keys()):
        ax[i].plot(data[mut].keys(), [i[0]/i[1] for i in data[mut].values()])
        ax[i].set_title(f"Percentage vs steps for {mut}")
        ax[i].set_xlabel("Number of mutation steps")
        ax[i].set_ylabel("Percentage of\n  watermarks detected")
        ax[i].set_ylim([0,1.2])
    fig.tight_layout(pad=5.0)
    plt.show()
    plt.savefig("mutator.png")
    
if __name__ == "__main__":
    eval()
