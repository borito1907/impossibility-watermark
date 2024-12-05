def clean_logfile(input_file: str, output_file: str = None):
    # Use the same file if no output file is specified
    if output_file is None:
        output_file = input_file

    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Filter out lines that start with 'Batches'
    cleaned_lines = [line for line in lines if not line.startswith('Batches')]

    with open(output_file, 'w') as f:
        f.writelines(cleaned_lines)

    print(f"Cleaned log saved to {output_file}")

# Example usage:
clean_logfile('/data2/borito1907/impossibility-watermark/10_23_adaptive_dev_gen.log', "/data2/borito1907/impossibility-watermark/10_23_adaptive_dev_gen_clean.log")
