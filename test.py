from mutators.llm import LLMMutator
from mutators.mask_fill import MaskFillMutator
from mutators.span_fill import SpanFillMutator
from oracle import SoloOracle, JointOracle, RelativeOracle, RankOracle
import csv

mutator = SpanFillMutator()

oracle = SoloOracle()
results = []
with open("./tests/mutator/tests_v1.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    for row in reader:
        print(row)
        _, prompt, response = row
        mutated = mutator.mutate(response)
        results.append(oracle.is_quality_preserved(prompt, response, mutated))