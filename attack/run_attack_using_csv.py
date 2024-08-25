from watermarker_factory import get_default_watermarker
from mutators.sentence import SentenceMutator
from mutators.span import SpanMutator
from mutators.word import WordMutator
from mutators.document_1step import DocumentMutator_1step
from mutators.document_2step import DocumentMutator_2step
from oracles import (
    SoloOracle, RankOracle, JointOracle, RelativeOracle,
    PrometheusAbsoluteOracle, PrometheusRelativeOracle, 
    BinaryOracle, MutationOracle, Mutation1Oracle, ExampleOracle, DiffOracle,
    ArmoRMOracle, InternLMOracle, OffsetBiasOracle
)
from attack import Attack

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
    "DocumentMutator_1step": DocumentMutator_1step,
    "DocumentMutator_2step": DocumentMutator_2step
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

@hydra.main(version_base=None, config_path="../conf", config_name="attack")
def main(cfg):
    watermarker = get_default_watermarker(cfg.watermarker)

    mutator = mutators[cfg.mutator_type]()
    oracle = oracles[cfg.oracle_type]()

    attacker = Attack(cfg, watermarker, mutator, oracle)

    dev_df = pd.read_csv('/local1/borito1907/impossibility-watermark/data/WQE/dev.csv')
    # watermarked_dev_df = pd.read_csv('/local1/borito1907/impossibility-watermark/human_study/data/dev_watermarked.csv')

    # prompt = dev_df.loc[dev_df[id] == 3949048841, 'prompt'].values[0]
    # watermarked_text = watermarked_dev_df.loc[watermarked_dev_df[id] == 3949048841, 'prompt'].values[0]

    attacked_text = attacker.attack(prompt, watermarked_text)

    log.info(f"Attacked Text: {attacked_text}")

if __name__ == "__main__":
    main()
