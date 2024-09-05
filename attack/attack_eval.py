from watermarker_factory import get_default_watermarker
from mutators import (
    SentenceMutator, SpanMutator, WordMutator, 
    DocumentMutator, DocumentMutator_1step, DocumentMutator_2step
)
from oracles import (
    SoloOracle, RankOracle, JointOracle, RelativeOracle,
    PrometheusAbsoluteOracle, PrometheusRelativeOracle, 
    BinaryOracle, MutationOracle, Mutation1Oracle, ExampleOracle, DiffOracle,
    ArmoRMOracle, InternLMOracle, OffsetBiasOracle
)
from attack.attack import Attack

import hydra
import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
logging.getLogger('optimum.gptq.quantizer').setLevel(logging.WARNING)

@hydra.main(version_base=None, config_path="../conf", config_name="attack")
def main(cfg):

    watermarkers = [
        "adaptive",
        "semstamp", 
        "umd"
    ]

    mutators = [
        SentenceMutator,
        SpanMutator,
        WordMutator,
        DocumentMutator,
        DocumentMutator_1step,
        DocumentMutator_2step
    ]

    # Step 1: Initialize Quality Oracle
    model_path = "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-IMP-0.1-q8_0.gguf"
    llm = models.LlamaCpp(
        model=model_path,
        echo=False,
        n_gpu_layers=-1,
        n_ctx=4096
    )
    oracle = DiffOracle(llm=llm, explain=False)

    for watermarker in watermarkers:

        # Step 2: Load data for that particular watermarker
        data = pd.read_csv(f"./data/WQE_{watermarker}/dev.csv")

        # Step 3: Initialize watermark detector
        watermarker = get_default_watermarker(watermarker)

        for mutator in mutators:

            # Step 4: Initialize Mutator
            mutator = mutator()

            # Step 5: Initialize Attacker
            o_str = oracle.__class__.__name__
            w_str = watermarker.__class__.__name__
            m_str = mutator.__class__.__name__
            cfg.log_csv_path = f"./attack_traces/{o_str}_{w_str}_{m_str}_attack_results.csv"
            
            attacker = Attack(cfg, mutator, oracle, watermarker)

            for benchmark_id, row in data.iterrows():

                out = {**row}
                out.update({"benchmark_id": benchmark_id})

                attacked_text = attacker.attack(prompt, watermarked_text)

                log.info(f"Attacked Text: {attacked_text}")

if __name__ == "__main__":
    main()
