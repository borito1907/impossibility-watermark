import subprocess
import logging
import sys
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

log = logging.getLogger(__name__)

def run_command(command, filepath):
    log.info(f"Running command: {command}")
    log.info(f"Saving results to {filepath}")

    try:
        with open(filepath, "w") as f:
            # Redirect both stdout and stderr to the same file
            _ = subprocess.run(command, shell=True, check=True, text=True, stdout=f, stderr=subprocess.STDOUT)
        log.info("Command executed successfully")
        return "Success", None
    except subprocess.CalledProcessError as e:
        log.error(f"Command failed with error: {e.stderr}")
        # Returning None, stderr if there was an error
        return None, e.stderr

# NOTE: pass in as an argument the prompt number (1, 2, or 3)
# this was so i could parallelize it better with gpus
def main():
    # TODO: Don't forget to change this. This should ideally, be a command line arg, but it's 11 pm. - Boran
    watermark_scheme = 'semstamp'

    dir_name = f'dev_{watermark_scheme}'
    base_folder_name = f'./inputs/{dir_name}'
    gen_stats_folder_path = f"{base_folder_name}/gen_stats"

    # Create the dirs if they don't exist.
    dirname = os.path.dirname(__file__)
    path1 = os.path.join(dirname, f'{gen_stats_folder_path}/')
    path2 = os.path.join(dirname, f'{base_folder_name}/logs/')
    paths = [path1, path2]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

    temp = 1
    divp = 0
    gen_type = "prompt"

    if gen_type not in ["prompt", "c4"]:
        raise Exception(f"Generation type {gen_type} is not supported.")


    for prompt_num in range(150,151):
        log_filepath = f"{base_folder_name}/logs/logfile_{prompt_num}.log"
                    
        command = f"python watermarked_text_generator.py " \
                f"++prompt_file='./data/WQE/dev.csv' " \
                f"++prompt_num={prompt_num} " \
                f"++is_completion=False " \
                f"++generator_args.temperature={temp} " \
                f"++generator_args.diversity_penalty={divp} " \
                f"++generation_stats_file_path='{gen_stats_folder_path}/prompt_{prompt_num}.csv' " \
                f"++watermark_args.use_fine_tuned=False " \
                f"++watermarked_text_file_path='{base_folder_name}/watermarked_texts.csv' "
    
        stdout, stderr = run_command(command, log_filepath)
        if stderr is None:
            print(f"Command succeeded: {command}\nOutput:\n{stdout}")
        else:
            print(f"Command failed: {command}\nError:\n{stderr}")

        log.info(f"stdout, stderr")
        log.info((stdout, stderr))

if __name__ == "__main__":
    main()