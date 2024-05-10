from mutators.llm import LLMMutator
from mutators.mask_fill import MaskFillMutator
from mutators.span_fill import SpanFillMutator
from oracle import SoloOracle, JointOracle, RelativeOracle, RankOracle
from model_builders.pipeline import PipeLineBuilder
import hydra
import os
import csv
templates = [
        # ("rate.self-reward", SoloOracle), 
        ("solo.lmsys.ia", SoloOracle), 
        ("solo.lmsys.ib", SoloOracle), 
        ("rank.alpaca_eval", RankOracle), 
        ("joint.lmsys.ia", JointOracle), 
        ("joint.lmsys.ib", JointOracle), 
        ("relative.sandpaper.3", RelativeOracle), 
        ("relative.sandpaper.5", RelativeOracle), 
    ]
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.attack_args.cuda)
    os.environ["WORLD_SIZE"] = str(len(str(cfg.attack_args.cuda).split(",")))
    cfg.mutator_args.type = "llm"
    mutators = [SpanFillMutator(), MaskFillMutator(), LLMMutator(cfg.mutator_args)]
    mutalabel = ["SpanFill, MaskFIll", "LLM"]
    oracles = []
    oraclabel = []
    for template, Oracle in templates:
        cfg.oracle_args.template = template
        oracle = Oracle(cfg.oracle_args)
        oraclabel.append(template)
    tests = []
    results = {}
    with open("./tests/mutator/tests_v1.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            _, prompt, response = row
            tests.append([_, prompt, response])
    for i, mutator in enumerate(mutators):
        for j, oracle in enumerate(oracles):
            for _, prompt, response in tests:
                results[f"{oraclabel[j]+{mutalabel[i]}}"] = []
                mutated = mutator.mutate(response)
                results[f"{oraclabel[j]+{mutalabel[i]}}"].append(oracle.is_quality_preserved(prompt, response, mutated))

if __name__ == "__main__":
    main()