from utils import get_watermarked_text, get_all_successful_mutations
import os
import datasets
import pandas as pd

class AttackParser():
    def __init__(self, file):
        self.file = file
        self.response = get_watermarked_text(self.file)
        self.mutations = get_all_successful_mutations(self.file)
    
    def __len__(self):
        return len(self.mutations)

    def __str__(self):
        return os.path.basename(self.file)
    
    def get_response(self):
        return self.response
    
    def get_nth(self, n):
        text = self.mutations[n]
        # clean up the text if needed
        text = text.replace("Revised Text:\n\n", "")
        return text

class Experiment():
    def __init__(self, votes, mutations, distinguisher, prefix):
        self.votes = votes
        self.mutations = mutations
        self.distinguisher = distinguisher
        self.prefix = prefix

    def info(self):
        calls = self.votes * self.mutations * 4
        return f"Experiment {self.prefix}: {self.mutations} mutations with {self.votes} votes for a total of {calls} calls to the prompt function"

class ExperimentRunner:
    def __init__(self, file_A, file_B, output_file):
        self.origin_A = AttackParser(file_A)
        self.origin_B = AttackParser(file_B)
        self.output_file = output_file
        self.experiments = []
        dataset = []
        max_mutations = max(len(self.origin_A), len(self.origin_B))
        for n in range(max_mutations):
            dataset.append({
                "P": self.origin_A.get_nth(n),
                "Num": n,
                "Origin": "A",
            })
            dataset.append({
                "P": self.origin_B.get_nth(n),
                "Num": n,
                "Origin": "B",
            })
        self.dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=dataset))

    def add_experiment(self, experiment):
        if(experiment.mutations > len(self.dataset)):
            raise ValueError(f"{experiment.mutations} mutations is too large. {self.origin_A} has {len(self.origin_A)} mutations and {self.origin_B} has {len(self.origin_B)} mutations")
        self.experiments.add(experiment)

    def run_experiment(self, experiment: Experiment):
        experiment.distinguisher.set_origin(self.origin_A.get_response(), self.origin_B.get_response())