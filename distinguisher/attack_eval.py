# RUN: CUDA_VISIBLE_DEVICES=4,5,6,7 python -m distinguisher.attack_eval

import traceback
import pandas as pd
from guidance import models 
from watermarker_factory import get_default_watermarker
from mutators import (
    SentenceMutator, SpanMutator, WordMutator, 
    DocumentMutator, Document1StepMutator, Document2StepMutator
)
from oracles import DiffOracle
from attack.attack import Attack

import hydra
import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
logging.getLogger('optimum.gptq.quantizer').setLevel(logging.WARNING)

@hydra.main(version_base=None, config_path="../conf/distinguisher", config_name="attack")
def main(cfg):

    watermarkers = [
        "adaptive",
        "umd"
    ]

    mutators = [
        WordMutator,
        SpanMutator,
        SentenceMutator,
        # DocumentMutator,
        # Document1StepMutator,
        # Document2StepMutator
    ]

    # Step 1: Initialize Quality Oracle
    model_path = "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-IMP-DiffOracle-0.1-q8_0.gguf"
    llm = models.LlamaCpp(
        model=model_path,
        echo=False,
        n_gpu_layers=-1,
        n_ctx=8192
    )
    oracle = DiffOracle(llm=llm, explain=False)

    for watermarker in watermarkers:

        # Step 2: Load data for that particular watermarker
        data = pd.read_csv(f"./distinguisher/responses/{watermarker}_responses.csv")

        for mutator in mutators:
            log.info("Initializing mutator...")
            # Step 4: Initialize Mutator
            mutator = mutator()

            for compare_against_original in [True]:

                # Step 5: Initialize Attacker
                o_str = oracle.__class__.__name__
                w_str = watermarker
                m_str = mutator.__class__.__name__
                cfg.attack.compare_against_original = compare_against_original
                cfg.attack.log_csv_path = f"./distinguisher/attack_traces/{o_str}_{w_str}_{m_str}_compare-original={compare_against_original}_attack_results.csv"
                
                log.info(f"Initializing attacker...")
                attacker = Attack(cfg, mutator, oracle)

                # Step 6: Attack each row in dataset
                for benchmark_id, row in data.iterrows():
                    try:
                        log.info(f"Attacking Row: {row}")
                        attacked_text = attacker.attack(row['prompt'], row['text'])
                        log.info(f"Attacked Text: {attacked_text}")
                    except Exception:
                        log.info(traceback.format_exc())

if __name__ == "__main__":
    main()
