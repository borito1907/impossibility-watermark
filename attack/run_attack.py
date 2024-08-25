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

    prompt = """Create a psychological portrait of a person who values their time 1000000000usd per second"""

    watermarked_text = """What an intriguing request! Let's dive into the psyche of someone who values their time at an astronomical rate of $1 billion per second.

**Name:** Temporis (a Latin-derived name that reflects their reverence for time)

**Personality:**

Temporis is an extremely goal-oriented individual, perpetually focused on maximizing productivity and efficiency. With a self-assessed value of their time at $1 billion per second, they treat every moment as a precious investment opportunity. This fixation drives them to prioritize tasks with ruthless pragmatism, dismissing any activity that doesn't offer substantial returns on their temporal capital.

**Traits:**

* **Time-consciousness**: Temporis has a hyper-acute sense of time, always aware of the ticking clock and the dollars flowing away with each passing second.
* **Opportunity cost obsession**: They constantly weigh the potential benefits of each action against its time expenditure, ensuring that every decision optimizes their schedule.
* **Frugality**: Given the staggering value they place on their time, Temporis tends to be incredibly thrifty, minimizing non-essential expenses and investments that don't generate significant ROI (return on investment).
* **High-stakes decision-making**: Every choice becomes a high-pressure situation, where even minor setbacks or miscalculations could result in billions lost."""

    attacked_text = attacker.attack(prompt, watermarked_text)

    log.info(f"Attacked Text: {attacked_text}")

if __name__ == "__main__":
    main()
