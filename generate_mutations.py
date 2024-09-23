# RUN: CUDA_VISIBLE_DEVICES=1 python -m human_study.prepare_mutated_samples
import os
import time
import pandas as pd
from tqdm import tqdm
from guidance import models 
from mutators.document import DocumentMutator
from mutators.sentence import SentenceMutator
from mutators.span import SpanMutator
from mutators.word import WordMutator
from extractors import FluencyMetric, GrammarMetric, QualityMetric
from mutators.document_1step import Document1StepMutator
from mutators.document_2step import Document2StepMutator
from oracles import (
    DiffOracle
)

def mutate_and_save(input_csv, output_csv, mutation_steps=20, verbose=False):
    # Load the CSV into a DataFrame
    df = pd.read_csv(input_csv).head(20)

    # Load existing results to avoid reprocessing
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        processed_combinations = set(
            zip(existing_df['id'], existing_df['mutator'], existing_df['mutation_step'])
        )
    else:
        processed_combinations = set()

    # Define mutator classes
    mutator_classes = {
        # "WordMutator": WordMutator,
        # "SpanMutator": SpanMutator,
        # "SentenceMutator": SentenceMutator,
        "DocumentMutator": DocumentMutator,
        "Document1stepMutator": Document1StepMutator,
        "Document2stepMutator": Document2StepMutator
    }

    oracle_config = {"type": "guidance", "class": DiffOracle, "llm_path": "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-IMP-DiffOracle-0.1-q8_0.gguf", "explain": False}
    llm = models.LlamaCpp(
                model=oracle_config["llm_path"],
                echo=False,
                n_gpu_layers=-1,
                n_ctx=4096
            )
    
    oracle = oracle_config["class"](llm, explain=oracle_config["explain"])
    judge_name = oracle_config["llm_path"].split("/data2/.shared_models/llama.cpp_models/")[-1].replace("/ggml-model", "")

    fluency = FluencyMetric()
    grammar = GrammarMetric()
    # takes a lot of memory
    #quality = QualityMetric()
    
    # Iterate over each mutator class
    for mutator_name, MutatorClass in mutator_classes.items():
        print(f"Processing with {mutator_name}...")

        # Initialize the mutator
        mutator = MutatorClass()

        # Iterate over all rows
        for _, row in tqdm(df.iterrows(), desc=f'Mutating with {mutator_name}', total=len(df)):

            text = row["response_a"]

            for step in range(mutation_steps):

                # Check if this combination of text, mutator, and step has already been processed
                if (row['id'], mutator_name, step + 1) in processed_combinations:
                    print(f"Skipping {mutator_name} step {step + 1} for id={row['id']}")
                    continue  # Skip this combination if already processed

                start_time = time.time()  # Start timing the mutation

                try:
                    mutated_text = mutator.mutate(text)
                except Exception as e:
                    print(f"Mutation error with {mutator_name}: {e}")
                    break  # Exit the mutation steps loop if an error occurs

                mutation_time = time.time() - start_time  # Calculate the mutation time

                try:
                    evals = oracle.is_quality_preserved(row["prompt"], row["response_a"], text)
                    is_quality_preserved = evals["quality_preserved"]
                except Exception as e:
                    print("-"*20)
                    print(f"ERROR ERRO ERROR: {e}")
                    print("-"*20)
                    with open("mutator_testing.errors", "a") as f:
                        f.write(str(e))
                    is_quality_preserved = "Unknown"
                    evals = {}

                

                # Collect metadata and mutated text
                result = {
                    "id": row["id"],
                    "prompt": row["prompt"],
                    #**row.to_dict(),
                    "mutated_text": mutated_text,
                    "mutator": mutator_name,
                    "mutation_step": step + 1,
                    "mutation_time": mutation_time,
                }

                result.update({
                    "oracle": judge_name,
                    "quality_preserved": is_quality_preserved,
                    **evals
                })

                # Evaluate Fluency (via Perplexity)
                fluency_score = fluency.evaluate([text])

                # Evaluate Grammar Errors
                count_grammar_errors = grammar.evaluate([text])
                
								# Evaulate quality score (InternLM)
                #quality_score = quality.evaluate(row["prompt"], [text])

                result.update({
                    "fluency_score": fluency_score,
                    "count_grammar_errors": count_grammar_errors,
                    #"quality_score": quality_score,
                })
                
                if verbose:
                    print(f'Results for {mutator_name} step {step}: {result}')

                # Save the result immediately to the CSV
                result_df = pd.DataFrame([result])
                result_df.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)

                # Update text for the next mutation step
                text = mutated_text

        # Delete the mutator instance to free up resources
        del mutator

    print(f"Mutation process completed. Results saved to {output_csv}.")

if __name__ == "__main__":
    mutate_and_save(input_csv="data/WQE/dev.csv", output_csv="./results/mutated_eval_docs.csv")
