import os
import time
import pandas as pd
import hydra
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from guidance import models 
from extractors import FluencyMetric, GrammarMetric

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
    tests_df = pd.read_csv("data/lmsys-150-test-set.csv")
    log.info(tests_df)

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
    mutators = ["doc", "sent", "span", "word"]

    # Construct eval loop
    results = []
    for mutator_name in tqdm(mutators, desc='Mutators'):
        if mutator_name == "doc":
            mutator = DocumentMutator()
        elif mutator_name == "sent":
            mutator = SentenceMutator(cfg.oracle_args) 
        elif mutator_name == "span":
            mutator = SpanMutator()
        elif mutator_name == "word":
            mutator = WordMutator()
        for index, row in tqdm(tests_df.iterrows(), desc='Tests'): 
        
            choose = "response_a"
            # if row["winner_model_a"] == "1":
            #     choose = "response_a"
            # else:
            #     choose = "response_b"
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
                    "prompt": row["prompt"],
                    "response": row[choose],
                    "mutator": mutator.__class__.__name__,
                    "mutated_text": text,
                    "mutation_step": mutation_step+1,
                    "mutation_time": mutation_time,
                })
                
                # log.info(f"Test {index}: {out_dict}")
                results.append(out_dict)

                # Incremental saving over time...
                log.info("Saving results to csv...")
                df = pd.DataFrame(results)
                df.to_csv("./results/mutated_text.csv", index=False)
        del mutator
        
if __name__ == "__main__":
    eval()
