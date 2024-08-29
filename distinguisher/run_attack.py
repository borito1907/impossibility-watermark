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
    ArmoRMOracle, InternLMOracle, OffsetBiasOracle, HumanOracle
)
from attack.attack import Attack

import hydra
import logging
import pandas as pd
from tqdm import tqdm

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
    "OffsetBiasOracle": OffsetBiasOracle,
    "HumanOracle": HumanOracle
}

@hydra.main(version_base=None, config_path="../conf", config_name="attack")
def main(cfg):
    df = pd.read_csv("./distinguisher/semstamp_responses.csv")
    df = df.loc[[0,1,2]]

    watermarker = None
    if cfg.attack.check_watermark:
        watermarker = get_default_watermarker(cfg.watermarker)
    mutator = mutators[cfg.mutator_type]()
    oracle = oracles[cfg.oracle_type]()

    attacker = Attack(cfg, mutator, oracle, watermarker)

    for i, row in tqdm(df.iterrows(), total=len(df), desc="ID progress"):
        cfg.attack.log_csv_path = f"./distinguisher/attack/word/human_{i}.csv"
        prompt = row["prompt"]
        watermarked_text = row["text"]
        attacked_text = attacker.attack(prompt, watermarked_text)

        log.info(f"Attacked Text: {attacked_text}")

if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=0 python -m distinguisher.run_attack