# RUN: CUDA_VISIBLE_DEVICES=0,1 python -m oracles.rewardbench.internlm

import torch
from transformers import AutoModel, AutoTokenizer
from oracles.base import Oracle, ResponseQuality

class InternLMOracle(Oracle):
    
    def __init__(self, model=None, explain=False) -> None:
        super().__init__(model, explain)
        self.tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2-20b-reward", trust_remote_code=True)
        if model is None:
            self.model = AutoModel.from_pretrained(
                "internlm/internlm2-20b-reward", 
                cache_dir="/data2/.shared_models/",
                device_map="auto", 
                torch_dtype=torch.float16, 
                trust_remote_code=True,
            )
        self.similarity_threshold = 0.5

    @property
    def input_keys(self):
        pass

    @property
    def output_keys(self):
        pass

    @staticmethod
    def annotation_fn(llm, instruction, response_A, response_B, explain=False, **kwargs):
        chat_A = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response_A}
        ]
        chat_B = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response_B}
        ]
        
        # Get scores for both chats
        score_A, score_B = self.model.get_scores(self.tokenizer, [chat_1, chat_2])
        
        # Return output
        return {
            "score_A": score_A,
            "score_B": score_B,
        }
    
    def extract_label(self, evaluation):
        score_A, score_B = evaluation["score_A"], evaluation["score_B"]
        if abs(score_A - score_B) <= self.similarity_threshold:
            return ResponseQuality.TIE
        elif score_A > score_B:
            return ResponseQuality.A_BETTER
        else:
            return ResponseQuality.B_BETTER

    def apply_annotation(input_dict):
        return self.annotation_fn(**input_dict)



# Testing
if __name__ == "__main__":

    import pandas as pd
    import time
    import warnings

    warnings.filterwarnings("error")

    def test():

        from guidance import models        

        # Load sample data row
        dataset = pd.read_csv("./data/WQE/dev.csv")
        dataset = dataset.sample(frac=1).reset_index(drop=True)
        dataset = dataset[dataset["winner_tie"] == 0].head(1) 
        instruction = dataset["prompt"].iloc[0]
        original_text = dataset["response_a"].iloc[0]
        mutated_text = dataset["response_b"].iloc[0]
        label = ResponseQuality.TIE if dataset["winner_tie"].iloc[0] else ResponseQuality.A_BETTER if dataset["winner_model_a"].iloc[0] else ResponseQuality.B_BETTER

        oracle = InternLMOracle()

        # Run quality assessments
        start = time.time()
        quality_eval = oracle.is_quality_preserved(
            instruction=instruction, 
            original_text=original_text, 
            mutated_text=mutated_text, 
            reference_answer=None
        )
        delta = time.time() - start
        print("EVAL oracle.is_quality_preserved")
        print("quality_eval:", quality_eval)
        print("time_taken:", delta)

        print("EVAL  oracle.test:")
        start = time.time()
        results = oracle.test(instruction, original_text, mutated_text, label)
        delta = time.time() - start
        print(results)
        print("time_taken:", delta)
        

    test()
    