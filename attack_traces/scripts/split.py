# RUN: python -m attack_traces.scripts.split

import pandas as pd
import numpy as np

def split_csv(input_file, output_file_base):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Split the data into 10 chunks
    chunks = np.array_split(df, 10)
    
    # Write each chunk to a new CSV file with numbers 1-10 appended
    for i, chunk in enumerate(chunks, 1):
        output_file = f"{output_file_base}{i}.csv"
        chunk.to_csv(output_file, index=False)
        print(f"Saved {output_file}")

# Usage
input_file = './attack_traces/DiffOracle_umd_new_WordMutator_n-steps=1000_attack_results.csv'
output_file_base = './attack_traces/DiffOracle_umd_new_WordMutator_n-steps=1000_attack_results'

split_csv(input_file, output_file_base)
