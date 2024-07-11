import os
import time
import pandas as pd
import hydra
from tqdm import tqdm
import logging
from oracles.absolute import PrometheusAbsoluteOracle
import matplotlib.pyplot as plt

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
    from mutators.word import MaskFillMutator
    from mutators.span import SpanFillMutator
    from mutators.dipper import DipperParaphraser
    # Set number of mutation steps to analyze
    mutation_steps = 10
    log.info(f"Setting number of mutation steps to {mutation_steps}...")

    # Load test data
    # NOTE: we will reuse the outputs from the quality oracle tests
    log.info("Loading tests...")
    tests_df = pd.read_csv("./tests/mutator/tests_v2.csv")
    log.info(tests_df)

    # Init shared pipeline for oracles and LLMMutator
    log.info("Initializing shared pipeline for oracles and LLMMutator...")
    # pipeline = PipeLineBuilder(cfg.oracle_args)

    # Init oracles
    # templates = [
    #     ("solo.lmsys.ib", SoloOracle), 
    #     # ("joint.lmsys.ib", JointOracle), 
    #     ("relative.sandpaper.3", RelativeOracle), 
    # ]
    # log.info(f"Initializing oracles: {','.join(t for t,c in templates)}...")
    prometheus = PrometheusAbsoluteOracle(cfg)
    oracles = []
    # for t, c in templates:
    #     cfg.oracle_args.template = t
    #     oracles.append(c(cfg=cfg.oracle_args, pipeline=pipeline))

    # Init mutators
    log.info(f"Initializing mutators: Document, Sentence, MaskFillMutator (ours), SpanFillMutator (sandpaper)...")
    s_mutator = SentenceMutator(cfg.oracle_args)
    dip_mutator = DipperParaphraser()
    mf_mutator = MaskFillMutator()
    sf_mutator = SpanFillMutator()
    mutators = [s_mutator, dip_mutator, mf_mutator, sf_mutator]

    # Construct eval loop
    results = []
    for index, row in tqdm(tests_df.iterrows(), desc='Tests'): 
        for mutator in tqdm(mutators, desc='Mutators'):
            if row["winner_model_a"] == "1":
                choose = "response_a"
            else:
                choose = "response_b"
            text = row[choose]

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
                
                    # Evaluate Mutation Quality
                try:
                    evals = prometheus.is_quality_preserved(row["prompt"], row[choose], text)
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
                    "oracle": "Prometheus: Relative",
                    "quality_preserved": is_quality_preserved,
                    **evals
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
