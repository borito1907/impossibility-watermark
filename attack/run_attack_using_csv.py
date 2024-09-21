from watermarker_factory import get_default_watermarker
from mutators import (
    SentenceMutator, SpanMutator, WordMutator, 
    DocumentMutator, Document1StepMutator, Document2StepMutator)
from oracles import (
    SoloOracle, RankOracle, JointOracle, RelativeOracle,
    BinaryOracle, MutationOracle, Mutation1Oracle, ExampleOracle, DiffOracle,
    ArmoRMOracle, InternLMOracle, OffsetBiasOracle
)
from attack.attack import Attack

import hydra
import logging
import pandas as pd

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
logging.getLogger('optimum.gptq.quantizer').setLevel(logging.WARNING)

mutators = {
    "SentenceMutator": SentenceMutator,
    "SpanMutator": SpanMutator,
    "WordMutator": WordMutator,
    "DocumentMutator": DocumentMutator,
    "Document1StepMutator": Document1StepMutator,
    "Document2StepMutator": Document2StepMutator
}

oracles = {
    "SoloOracle": SoloOracle,
    "RankOracle": RankOracle,
    "JointOracle": JointOracle,
    "RelativeOracle": RelativeOracle,
    "BinaryOracle": BinaryOracle,
    "MutationOracle": MutationOracle,
    "Mutation1Oracle": Mutation1Oracle,
    "ExampleOracle": ExampleOracle,
    "DiffOracle": DiffOracle,
    "ArmoRMOracle": ArmoRMOracle,
    "InternLMOracle": InternLMOracle,
    "OffsetBiasOracle": OffsetBiasOracle
}

@hydra.main(version_base=None, config_path="../conf", config_name="attack")
def main(cfg):
    # dev_df = pd.read_csv('/local1/borito1907/impossibility-watermark/data/WQE/dev.csv')
    watermarked_dev_df = pd.read_csv('human_study/data/wqe_watermark_samples.csv')
    
    watermarked_dev_df = watermarked_dev_df.sample(frac=1).reset_index(drop=True).head(1)
    prompt = watermarked_dev_df.iloc[0]["prompt"]
    watermarked_text = watermarked_dev_df.iloc[0]["text"]
    watermarker_scheme = watermarked_dev_df.iloc[0]["watermarking_scheme"]
    
    watermarker = get_default_watermarker(watermarker_scheme)
    mutator = mutators[cfg.mutator_type]()
    oracle = oracles[cfg.oracle_type]()

    attacker = Attack(cfg, mutator, quality_oracle=oracle, watermarker=watermarker)

    attacked_text = attacker.attack(prompt, watermarked_text)

    log.info(f"Attacked Text: {attacked_text}")

if __name__ == "__main__":
    main()
