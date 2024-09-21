from mutators.sentence import SentenceMutator
from mutators.span import SpanMutator
from mutators.word import WordMutator
from mutators.document_1step import Document1StepMutator
from mutators.document_2step import Document2StepMutator

from oracles import (
    SoloOracle, RankOracle, JointOracle, RelativeOracle,
    PrometheusAbsoluteOracle, PrometheusRelativeOracle, 
    BinaryOracle, MutationOracle, Mutation1Oracle, ExampleOracle, DiffOracle,
    ArmoRMOracle, InternLMOracle, OffsetBiasOracle
)

from attack import Attack

import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
logging.getLogger('optimum.gptq.quantizer').setLevel(logging.WARNING)

from watermarker_factory import get_default_watermarker

mutators = {
    "SentenceMutator": SentenceMutator,
    "SpanMutator": SpanMutator,
    "WordMutator": WordMutator,
    "Document1StepMutator": Document1StepMutator,
    "Document2StepMutator": Document2StepMutator
}

oracles = {
    "SoloOracle": SoloOracle,
    "RankOracle": RankOracle,
    "JointOracle": JointOracle,
    "RelativeOracle": RelativeOracle,
    "PrometheusAbsoluteOracle": PrometheusAbsoluteOracle,
    "PrometheusRelativeOracle": PrometheusRelativeOracle,
    "BinaryOracle": BinaryOracle,
    "MutationOracle": MutationOracle,
    "Mutation1Oracle": Mutation1Oracle,
    "ExampleOracle": ExampleOracle,
    "DiffOracle": DiffOracle,
    "ArmoRMOracle": ArmoRMOracle,
    "InternLMOracle": InternLMOracle,
    "OffsetBiasOracle": OffsetBiasOracle
}

import hydra

@hydra.main(version_base=None, config_path="conf", config_name="new_config")
def main(cfg):
    watermarker = get_default_watermarker(cfg.watermarker)

    mutator = mutators[cfg.mutator_type]()
    oracle = oracles[cfg.oracle_type]()

    attacker = Attack(cfg, watermarker, mutator, oracle)

    

    attacked_text = attacker.attack(prompt , watermarked_text)

    log.info(f"Attacked Text: {attacked_text}")

if __name__ == "__main__":
    main()
