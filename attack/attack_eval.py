# RUN: CUDA_VISIBLE_DEVICES=0,1,2 python -m attack.attack_eval

import os
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
import os
import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
logging.getLogger('optimum.gptq.quantizer').setLevel(logging.WARNING)

@hydra.main(version_base=None, config_path="../conf", config_name="attack")
def main(cfg):

    cfg.attack.check_watermark = False

    watermarkers = [
        # "umd",
        # "umd_new",
        # "unigram",
        # "semstamp", 
        "adaptive",
    ]

    mutators = [
        # WordMutator,
        # SpanMutator,
        # SentenceMutator,
        # Document1StepMutator,
        # Document2StepMutator,
        DocumentMutator,
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

        # Step 2: Load data for that particular watermarkerp
        # data = pd.read_csv('/data2/borito1907/impossibility-watermark/10_03_less_high_semstamp_good_embedder.csv')
        data = pd.read_csv(f"./data/WQE_{watermarker}/dev.csv")
        # data = pd.read_csv(f"./data/WQE_{watermarker}/dev.csv").sample(n=10, random_state=42)

        # Step 3: Initialize watermark detector
        # watermarker = get_default_watermarker(watermarker)
        watermarker_obj = None

        for mutator in mutators:
            log.info("Initializing mutator...")
            # Step 4: Initialize Mutator
            mutator = mutator()

            # Step 5: Initialize Attacker
            o_str = oracle.__class__.__name__
            # w_str = watermarker_obj.__class__.__name__
            m_str = mutator.__class__.__name__
            cfg.attack.compare_against_original = True

            if m_str == "WordMutator":
                cfg.attack.max_steps = 1000
            if m_str == "SpanMutator":
                cfg.attack.max_steps = 200
            if m_str == "SentenceMutator":
                cfg.attack.max_steps = 100
            if m_str == "DocumentMutator":
                cfg.attack.max_steps = 50

            cfg.attack.log_csv_path = f"./attack_traces/{o_str}_{watermarker}_{m_str}_n-steps={cfg.attack.max_steps}_attack_results.csv"
            # cfg.attack.log_csv_path = f"./attack_traces/good_embedder_{o_str}_{watermarker}_{m_str}_n-steps={cfg.attack.max_steps}_attack_results.csv"

            if os.path.exists(cfg.attack.log_csv_path):
                log.info(f"skipping this attack configuration: {cfg.attack.log_csv_path}")
                continue

            log.info(f"Initializing attacker...")
            attacker = Attack(cfg, mutator, oracle, watermarker_obj)

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
