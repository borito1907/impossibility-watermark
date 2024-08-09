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
    from mutators.document_1step import DocumentMutator_1step
    from mutators.document_2step import DocumentMutator_2step
    from mutators.sentence import SentenceMutator
    from mutators.span import SpanMutator
    from mutators.word import WordMutator
    # Set number of mutation steps to analyze
    mutation_steps = 20
    log.info(f"Setting number of mutation steps to {mutation_steps}...")

    # Load test data
    # NOTE: we will reuse the outputs from the quality oracle tests
    log.info("Loading tests...")
    tests_df = pd.read_csv("data/wqe_watermark_samples_converted.csv")
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
    # mutators = [ "span", "word", "doc", "doc_1", "doc_2", "sent"]
    mutators = ["doc", "sent", "doc_1", "doc_2"]

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
        elif mutator_name == "doc_1":
            mutator = DocumentMutator_1step(cfg.mutator_args)
        elif mutator_name == "doc_2":
            mutator = DocumentMutator_2step(cfg.mutator_args)
        for index, row in tqdm(tests_df.iterrows(), desc='Tests'): 
            text = row["text"]
            out_dict = {}
            out_dict.update(row)
            for mutation_step in range(mutation_steps):

                # Initialize output_dict
                

                # Mutate text
                start = time.time()
                # print(text)
                # print(type(text))
                try:
                    text = mutator.mutate(text)
                    if mutator_name == "sent":
                        text = text['mutated_text']
                except Exception as e:
                    print("-"*20)
                    print(f"ERROR ERRO ERROR: {e}")
                    with open("mutator_testing.errors", "a") as f:
                        f.write(str(e))
                    print("-"*20)
                    continue

                mutation_time = time.time() - start
                out_dict.update({
                    f"mutation_step{mutation_step + 1}": text,
                    "time": mutation_time,
                    "mutator": mutator.__class__.__name__
                })
                # log.info(f"Test {index}: {out_dict}")
            results.append(out_dict)

            # Incremental saving over time...
            log.info("Saving results to csv...")
            df = pd.DataFrame(results)
            df.to_csv("./results/mutated_text4.csv", index=False)
        del mutator
        
if __name__ == "__main__":
    eval()
