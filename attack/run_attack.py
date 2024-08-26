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
from attack.attack import Attack

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

    watermarked_text = """What an fascinating individual! Based solely upon valuing time at such an astronomical rate, I'll create an educated portrait of someone I'll call... Chrono (a name that reflects an affinity and obsession over time).

**Personification:**
 Chrono is someone who lives and breaths time management and prioritization. Their every thought, decision, and move is calculated to optimize each passing microsecond for ultimate utility. This person has transcended mere mortal concepts like ""leisure"" or ""dawdling."" For them, time is both precious and perilously scarce.

**Key Traits:**

1. **Frugality with Time**: Chrono is extremely stingy with his or her own time, as it's worth $1 billion every tick of one second (approximately $31."""

    attacked_text = attacker.attack(prompt, watermarked_text)

    log.info(f"Attacked Text: {attacked_text}")

if __name__ == "__main__":
    main()
