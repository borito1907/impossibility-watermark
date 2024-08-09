# RUN: CUDA_VISIBLE_DEVICES=0 python -m human_study.prepare_mutated_samples
import os
import time
import pandas as pd
from tqdm import tqdm
from mutators.document import DocumentMutator
from mutators.sentence import SentenceMutator
from mutators.span import SpanMutator
from mutators.word import WordMutator

def mutate_and_save(input_csv, output_csv, mutation_steps=20, verbose=False):
    # Load the CSV into a DataFrame
    df = pd.read_csv(input_csv)

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
        "WordMutator": WordMutator,
        "SpanMutator": SpanMutator,
        "SentenceMutator": SentenceMutator,
        "DocumentMutator": DocumentMutator,
    }

    # Iterate over each mutator class
    for mutator_name, MutatorClass in mutator_classes.items():
        print(f"Processing with {mutator_name}...")

        # Initialize the mutator
        mutator = MutatorClass()

        # Iterate over all rows
        for _, row in tqdm(df.iterrows(), desc=f'Mutating with {mutator_name}', total=len(df)):
            for step in range(mutation_steps):

                # Check if this combination of text, mutator, and step has already been processed
                if (row['id'], mutator_name, step + 1) in processed_combinations:
                    print(f"Skipping {mutator_name} step {step + 1} for id={row['id']}")
                    continue  # Skip this combination if already processed

                start_time = time.time()  # Start timing the mutation

                try:
                    mutated_text = mutator.mutate(row['text'])
                except Exception as e:
                    print(f"Mutation error with {mutator_name}: {e}")
                    break  # Exit the mutation steps loop if an error occurs

                mutation_time = time.time() - start_time  # Calculate the mutation time

                # Collect metadata and mutated text
                result = {
                    **row.to_dict(),
                    "mutated_text": mutated_text,
                    "mutator": mutator_name,
                    "mutation_step": step + 1,
                    "mutation_time": mutation_time,
                }

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
    mutate_and_save(input_csv="./human_study/data/wqe_watermark_samples.csv", output_csv="./human_study/data/wqe_watermark_samples_mutated.csv")
