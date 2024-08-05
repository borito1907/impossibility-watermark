from guidance import models
from distinguisher.models import (AggressiveSimple)
import pandas as pd
import os
import datasets
from utils import get_prompt_or_output, get_watermarked_text, get_nth_successful_perturbation
from dotenv import load_dotenv, find_dotenv

class AttackParser():
    def __init__(self, file):
        self.file = file
        self.response = get_watermarked_text(self.file)
    
    def get_response(self):
        return self.response
    
    def get_nth(self, n):
        text = get_nth_successful_perturbation(self.file, n)
        # clean up the text
        text = text.replace("Revised Text:\n\n", "")
        return text
        
    
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

response_A = AttackParser(get_file(6, 1, 3))
response_B = AttackParser(get_file(6, 2, 4))

trials = [
    {"votes": 3, "mutations": 10, "class": AggressiveSimple, "origin_A": response_A, "origin_B": response_B, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-70B-Instruct-q8_0.gguf"},
]

for config in trials:
    if "llama" in config["llm_path"]:
        llm = models.LlamaCpp(
                model=config["llm_path"],
                echo=False,
                n_gpu_layers=-1,
                n_ctx=2048
            )
        sd = config["class"](llm, distinguisher_persona, origin_A.get_response(), origin_B.get_response())
        origin_A = config["origin_A"]
        origin_B = config["origin_B"]
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
            dataset = sd.distinguish(dataset, "llama_")
        else:
            dataset = sd.distinguish_majority(dataset, config["votes"], "llama_")

df = dataset.to_pandas()
df["Response_A"] = response_A.get_response()
df["Response_B"] = response_B.get_response()
df.to_csv("./distinguisher/results/llama_cpp_test.csv")

# ./impossibility-watermark> CUDA_VISIBLE_DEVICES=0 python -m distinguisher.evaluate