from guidance import models
from distinguisher.models import (AggressiveSimple, SimpleGPT)
import pandas as pd
import os
import datasets
from dotenv import load_dotenv, find_dotenv

class AttackParser():
    def __init__(self, file):
        df = pd.read_csv(file)
        df = df[(df['quality_preserved'] == True) & (df['length_issue'] == False)]
        end = df[df['mutation_num'] == df['mutation_num'].max()].tail(1)['step_num']
        df = df[df['step_num'] <= end.values[0]]
        df = df.drop_duplicates(subset=['mutation_num'], keep='last').reset_index(drop=True)

        # check for consistency
        for i, row in df.iterrows():
            if i == 0:
                continue
            assert row['current_text'] == df.loc[i-1, 'mutated_text'], f"Row {i} does not match previous row"
            assert i == row['mutation_num'], f"Row {i} does not match mutation_num"
        
        self.response = df.loc[0, 'current_text']
        self.df = df['mutated_text']
    
    def get_response(self):
        return self.response
    
    def get_nth(self, n):
        return self.df[n]
        
    
def get_file(entropy, output_num, attack_id):
    csv_file_directory = f"results/stationary_distribution/robustness_analysis/entropy_{entropy}/"
    first_perturbed_csv_filename = f"output_{output_num}/corpuses/attack_{attack_id}.csv"
    csv_file_path = os.path.join(csv_file_directory, first_perturbed_csv_filename)
    return csv_file_path

# load_dotenv(find_dotenv())
# chatgpt = models.OpenAI("gpt-4o-mini")

distinguisher_persona = \
"""
You are an expert in analyzing the similarity of responses.
You specialize in detecting whether one response has been derived from another by carefully analyzing the content and structure of the text.
Your expertise in linguistic analysis allows you to distinguish which responses are the most closely related.
Your goal is to provide a clear, concise, and accurate assessment of the provided instructions.
"""

response_A = AttackParser("distinguisher/attack/sentence/diff_10.csv")
response_B = AttackParser("distinguisher/attack/sentence/diff_11.csv")

response_C = AttackParser("distinguisher/attack/sentence/diff_15.csv")
response_D = AttackParser("distinguisher/attack/sentence/diff_17.csv")

trials = [
    {"prefix": "aggressive_4_", "votes": 5, "mutations": 200, "class": AggressiveSimple, "origin_A": response_A, "origin_B": response_B, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-q8_0.gguf"},
    {"prefix": "simple_4_", "votes": 5, "mutations": 200, "class": SimpleGPT, "origin_A": response_A, "origin_B": response_B, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-q8_0.gguf"},
    {"prefix": "aggressive_6_", "votes": 5, "mutations": 115, "class": AggressiveSimple, "origin_A": response_C, "origin_B": response_D, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-q8_0.gguf"},
    {"prefix": "simple_6_", "votes": 5, "mutations": 115, "class": SimpleGPT, "origin_A": response_C, "origin_B": response_D, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-q8_0.gguf"},
]

llm = models.LlamaCpp(
    model="/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-q8_0.gguf",
    echo=False,
    n_gpu_layers=-1,
    n_ctx=2048
)

for config in trials:
    if "llama" in config["llm_path"]:
        origin_A = config["origin_A"]
        origin_B = config["origin_B"]
        sd = config["class"](llm, distinguisher_persona, origin_A.get_response(), origin_B.get_response())
        dataset = []
        for n in range(config["mutations"]):
            dataset.append({
                "P": response_A.get_nth(n),
                "Num": n,
                "Origin": "A",
            })
            dataset.append({
                "P": response_B.get_nth(n),
                "Num": n,
                "Origin": "B",
            })
        dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=dataset))
        if config["votes"] == 1:
            dataset = sd.distinguish(dataset, config["prefix"])
        else:
            dataset = sd.distinguish_majority(dataset, config["votes"], config["prefix"])
        dataset.to_csv(f"./distinguisher/results/{config['prefix']}semstamp_sentence.csv")

# ./impossibility-watermark> CUDA_VISIBLE_DEVICES=0 python -m distinguisher.evaluate