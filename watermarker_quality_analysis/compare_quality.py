import pandas as pd
import time
import warnings
import logging
from guidance import models

from oracles import (
    SoloOracle, RankOracle, JointOracle, RelativeOracle,
    PrometheusAbsoluteOracle, PrometheusRelativeOracle, 
    BinaryOracle, MutationOracle, Mutation1Oracle, ExampleOracle, DiffOracle,
    ArmoRMOracle, InternLMOracle, OffsetBiasOracle
)

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
logging.getLogger('optimum.gptq.quantizer').setLevel(logging.WARNING)

warnings.filterwarnings("error")

def main():
    oracle = OffsetBiasOracle()

    # Load the DFs TODO:
    
    # Loop over the same entries TODO:

    # Run quality assessments
    start = time.time()

    quality_eval = oracle.is_quality_preserved(
        instruction=instruction, 
        original_text=original_text, 
        mutated_text=mutated_text, 
        reference_answer=None
    )

    delta = time.time() - start
    log.info("EVAL oracle.is_quality_preserved")
    log.info("quality_eval:", quality_eval)
    log.info("time_taken:", delta)



if __name__ == "__main__":
    main()