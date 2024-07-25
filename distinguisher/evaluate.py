from guidance import models
from distinguisher.models.aggressive import AggressiveSimple
from distinguisher.models.simple import SimpleGPT
from distinguisher.models.reasoning import ReasoningDistinguisher, ReasoningGPT
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

model_id = "TechxGenus/Meta-Llama-3-8B-Instruct-GPTQ"

# Load the model
llm = models.Transformers(
    model_id, 
    echo=False,
    cache_dir="/data2/.shared_models/", 
    device_map='auto',
)

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
# response_A = AttackParser(get_file(4, 1, "2_1"))
# response_B = AttackParser(get_file(4, 2, "1_1"))
# sd = ReasoningGPT(chatgpt, distinguisher_persona, response_A.get_response(), response_B.get_response())
sd2 = AggressiveSimple(llm, distinguisher_persona, response_A.get_response(), response_B.get_response())

dataset = []
for n in range(2):
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
# dataset = sd.distinguish_majority(dataset, 5, "gpt_")
dataset = sd2.distinguish_majority(dataset, 3, "llama_")
# dataset = sd2.distinguish(dataset, "llama_")

df = dataset.to_pandas()
df["Response_A"] = response_A.get_response()
df["Response_B"] = response_B.get_response()
df.to_csv("./distinguisher/results/multi_reason_gpt.csv")

# ./impossibility-watermark> CUDA_VISIBLE_DEVICES=0 python -m distinguisher.evaluate