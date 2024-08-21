# RUN: CUDA_VISIBLE_DEVICES=0,1,2,3 python -m distinguisher.attack
import os
import time
import pandas as pd
from tqdm import tqdm
from mutators.sentence import SentenceMutator
from mutators.span import SpanMutator
from mutators.word import WordMutator
from oracles import DiffOracle
from guidance import models        

def mutate_and_save_with_oracle(input_csv, output_csv, verbose=False):
    # Load the CSV into a DataFrame
    df = pd.read_csv(input_csv)

    # Load existing results to avoid reprocessing
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        existing_df = existing_df[existing_df['quality_preserved'] == True]
        processed_combinations = set(
            zip(existing_df['id'], existing_df['mutator'], existing_df['step'])
        )
    else:
        processed_combinations = set()

    # Define experiment parameters
    class Experiment:
        def __init__(self, mutator_name, mutator_class, mutation_steps, compare_versus):
            self.mutator_name = mutator_name
            self.mutator_class = mutator_class
            self.mutation_steps = mutation_steps
            self.compare_versus = compare_versus

    experiment_params = [
        Experiment("SentenceMutator", SentenceMutator, 500, "origin"),
        Experiment("SentenceMutator", SentenceMutator, 100, "last"),
        Experiment("WordMutator", WordMutator, 500, "origin"),
        Experiment("WordMutator", WordMutator, 100, "last"),
        Experiment("SpanMutator", SpanMutator, 500, "origin"),
        Experiment("SpanMutator", SpanMutator, 100, "last"),
    ]

    model_id = "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-q8_0.gguf"
    llm = models.LlamaCpp(
        model=model_id,
        echo=False,
        n_gpu_layers=-1,
        n_ctx=2048
    )

    oracle = DiffOracle(llm, explain=False)

    # Iterate over each mutator class
    total_iterations = len(df) * sum(exp.mutation_steps for exp in experiment_params)
    with tqdm(total=total_iterations, desc="") as pbar:
        for i, exp in enumerate(experiment_params):
            pbar.set_description(f"{i+1}/{len(experiment_params)}: Initializing {exp.mutator_name}")
            mutator_name = exp.mutator_name
            MutatorClass = exp.mutator_class
            mutation_steps = exp.mutation_steps
            compare_versus = exp.compare_versus
            print(f"Processing with {mutator_name}...")

            # Initialize the mutator
            mutator = MutatorClass()

            # Iterate over all rows
            for _, row in df.iterrows():
                text = row["text"]

                step = 0
                patience = 20
                while step < mutation_steps: 
                    pbar.set_description(f"{i+1}/{len(experiment_params)}: {exp.mutator_name} step {step + 1}/{mutation_steps} for row {row['id']+1}/{len(df)}")
                    # Check if this combination of text, mutator, and step has already been processed
                    if (row['id'], mutator_name, step + 1) in processed_combinations:
                        print(f"Skipping {mutator_name} step {step + 1} for id={row['id']}", flush=True)
                        text = existing_df[(existing_df['id'] == row['id']) & (existing_df['mutator'] == mutator_name) & (existing_df['step'] == step + 1)]['mutated_text'].values[0]
                        step += 1
                        pbar.update(1)
                        continue  # Skip this combination if already processed

                    patience -= 1
                    if(patience == 0):
                        print(f"Patience expired for {mutator_name} step {step + 1} for id={row['id']}", flush=True)
                        pbar.update(mutation_steps - step)
                        break
                    start_time = time.time()  # Start timing the mutation

                    try:
                        mutated_text = mutator.mutate(text)
                    except Exception as e:
                        print(f"Mutation error with {mutator_name}: {e}")
                        continue # Retry if an error occurs

                    mutation_time = time.time() - start_time  # Calculate the mutation time

                    # Check if the mutation is valid
                    start_time = time.time()  # Start timing the oracle
                    try:
                        quality_preserved = oracle.is_quality_preserved(
                            instruction=row['prompt'], 
                            original_text=row['text'] if compare_versus=="origin" else text,
                            mutated_text=mutated_text, 
                            reference_answer=None
                        )['quality_preserved']
                    except Exception as e:
                        print(f"Oracle error with {mutator_name}: {e}")
                        continue # Retry if an error occurs
                    oracle_time = time.time() - start_time  # Calculate the oracle time

                    # Collect metadata and mutated text
                    result = {
                        **row.to_dict(),
                        "mutated_text": mutated_text,
                        "mutator": mutator_name,
                        "step": step + 1,
                        "mutation_time": mutation_time,
                        "oracle_time": oracle_time,
                        "quality_preserved": quality_preserved,
                        "compare_versus": compare_versus
                    }

                    if verbose:
                        print(f'Results for {mutator_name} step {step}: {result}')

                    # Save the result immediately to the CSV
                    result_df = pd.DataFrame([result])
                    result_df.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)

                    # Update text for the next mutation step
                    if(quality_preserved):
                        step += 1
                        pbar.update(1)
                        text = mutated_text
                        patience = 20

            # Delete the mutator instance to free up resources
            del mutator

    print(f"Mutation process completed. Results saved to {output_csv}.")

if __name__ == "__main__":
    mutate_and_save_with_oracle(input_csv="./distinguisher/watermarked_responses.csv", output_csv="./distinguisher/mutations.csv")
