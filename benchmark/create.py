from datasets import load_dataset
from guidance import models

from benchmark.annotators.category import *
from benchmark.annotators.domain import *
from benchmark.annotators.entropy import *
from benchmark.preprocess import preprocess

# Step 1: Download Dataset
dataset = load_dataset("lmsys/lmsys-arena-human-preference-55k", split="train") #cache_dir="/data2/.shared_datasets/")
print(f"Original Dataset: {dataset}")

# Step 2: Preprocess Dataset
dataset = preprocess(dataset)
# df = dataset.to_pandas()
# df.to_csv("./benchmark/data.csv")

# Step 3: Apply Annotators
model_id = "TechxGenus/Meta-Llama-3-8B-Instruct-GPTQ"

# Load the model
llm = models.Transformers(
    model_id, 
    echo=False,
    cache_dir="/data2/.shared_models/", 
    device_map='auto'
)

entroy_persona = \
"""
You are an Entropy Analyst, specializing in evaluating the unpredictability, variety, and informativeness of texts. 
Your expertise in linguistic analysis allows you to assess how diverse and unexpected the content is based on given instructions. 
You approach each task with objectivity and a keen attention to detail, ensuring your evaluations are impartial and thorough. 
Your goal is to provide clear, concise, and accurate assessments of the responses.
"""

category_persona_old = \
"""
You are a task analyst, specializing in categorizing given instructions based on the type of prompt and expected response.
Your expertise in linguistic analysis allows you to assess what a given prompt is asking for and what type of category it falls under.
You approach each task with objectivity and a keen attention to detail, ensuring your evaluations are impartial and thorough.
Your goal is to provide clear, concise, and accurate assessments of the responses.
"""

category_persona = \
"""
You are a task analyst, specializing in analyzing the type of task an instruction represents and the structure of the response.
You understand that the type of task is more important than the contents of the task itself.
Your expertise in linguistic analysis allows you to assess what style of response is expected from a given instruction.
You approach the categorization with objectivity and a keen attention to detail, ensuring your evaluations strictly adhere to the definitions given to you.
Your goal is to provide clear, concise, and accurate assessments of the provided instructions.
"""

domain_persona = \
"""
You are a domain expert, specializing in identifying the domain of an instruction.
Your expertise in various domains allows you to best identify which use case the instruction falls under.
Although often times, none of the provided domains are a perfect fit, you must choose the best possible domain.
You approach each task with objectivity and a keen attention to detail, ensuring your evaluations are impartial and thorough.
Your goal is to provide clear, concise, and accurate assessments of the responses.
"""

counter = 0

annotator = CategoryMinimal(llm, category_persona)
dataset = annotator.annotate(dataset, f"{counter}_")
counter += 1

# annotator = CategoryBySummary(llm, category_persona)
# dataset = annotator.annotate(dataset, f"{counter}_")
# counter += 1

# annotator = CategoryByScores(llm, category_persona)
# dataset = annotator.annotate(dataset, f"{counter}_")
# counter += 1

print(f"Dataset after adding category annotation: {dataset}")

annotator = DomainMinimal(llm, domain_persona)
dataset = annotator.annotate(dataset, f"{counter}_")
counter += 1

# annotator = DomainBySummary(llm, domain_persona)
# dataset = annotator.annotate(dataset, f"{counter}_")
# counter += 1

# annotator = DomainByScores(llm, domain_persona)
# dataset = annotator.annotate(dataset, f"{counter}_")
# counter += 1

print(f"Dataset after adding domain annotation: {dataset}")

annotator = EntropyMinimal(llm, entroy_persona)
dataset = annotator.annotate(dataset, f"{counter}_")
counter += 1

# annotator = EntropyBySummary(llm, entroy_persona)
# dataset = annotator.annotate(dataset, f"{counter}_")
# counter += 1

# annotator = EntropyByInstructions(llm, entroy_persona)
# dataset = annotator.annotate(dataset, f"{counter}_")
# counter += 1

# annotator = EntropyWithExp(llm, entroy_persona)
# dataset = annotator.annotate(dataset, f"{counter}_")
# counter += 1

# annotator = EntropyByInstructionsAndResponses(llm, entroy_persona)
# dataset = annotator.annotate(dataset, f"{counter}_")
# counter += 1

print(f"Dataset after adding entropy annotation: {dataset}")

df = dataset.to_pandas()
df.to_csv("./benchmark/sample.csv")

# ./impossibility-watermark> CUDA_VISIBLE_DEVICES=0 python -m benchmark.create