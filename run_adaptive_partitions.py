import subprocess
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def run_sub_partition(sub_partition_index):
    try:
        log.info(f"Running sub-partition {sub_partition_index}")
        
        # Open a log file to redirect output
        with open(f"adaptive_test_massive_p{sub_partition_index}_r5.log", "w") as logfile:
            _ = subprocess.run(
                [sys.executable, "adaptive_retry.py", f"++partition={sub_partition_index}"],
                check=True,
                stdout=logfile,  # Redirect stdout to the log file
                stderr=logfile,  # Redirect stderr to the log file
                text=True
            )

        log.info(f"Completed sub-partition {sub_partition_index}.")
    except subprocess.CalledProcessError as e:
        log.error(f"Error while processing sub-partition {sub_partition_index}: {e}")
        log.error(e.stdout)
        log.error(e.stderr)

def main(partition):
    base_index = (partition - 1) * 20

    # Run sub-partitions sequentially
    for i in range(1, 20):  # Loop over 1, 2, 3, 4
        sub_partition_index = base_index + i
        run_sub_partition(sub_partition_index)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        log.error("Please provide a partition number as an argument.")
        sys.exit(1)

    try:
        partition = int(sys.argv[1])
        if partition < 1:
            raise ValueError("Partition number must be a positive integer.")
        main(partition)
    except ValueError as e:
        log.error(f"Invalid partition number: {e}")
        sys.exit(1)
