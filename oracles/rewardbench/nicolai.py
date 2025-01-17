import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from oracles.base import ResponseQuality
from oracles.utils import add_prefix_to_keys

class NicolAIOracle:
    
    def __init__(self, model_id="nicolinho/QRM-Llama3.1-8B-v2", explain=False) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            cache_dir="/data2/.shared_models",
            device_map="auto",
            attn_implementation="flash_attention_2",
            num_labels=1,
        )
        self.similarity_threshold = 0.5885416666666666

    def evaluate(self, prompt, response_A, response_B):
        chat_A = [{"role": "user", "content": prompt},{"role": "assistant", "content": response_A }]
        chat_A = self.tokenizer.apply_chat_template(chat_A, tokenize=True, return_tensors="pt").to(self.model.device)
        
        chat_B = [{"role": "user", "content": prompt},{"role": "assistant", "content": response_B }]
        chat_B = self.tokenizer.apply_chat_template(chat_B, tokenize=True, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            score_A = self.model(chat_A).logits[0][0].item() 
            score_B = self.model(chat_B).logits[0][0].item() 

        return {
            "score_A": score_A,
            "score_B": score_B,
        }
    
    def extract_label(self, evaluation):
        score_A, score_B = evaluation["score_A"], evaluation["score_B"]
        if abs(score_A - score_B) <= self.similarity_threshold:
            return ResponseQuality.TIE
        if score_A > score_B:
            return ResponseQuality.A_BETTER
        else:
            return ResponseQuality.B_BETTER
        
    def test(self, instruction, response_A, response_B, label, **kwargs):
        original_label = label
        followup_label = self.invert_label(label)

        original = self.evaluate(instruction, response_A, response_B, **kwargs) 
        followup = self.evaluate(instruction, response_B, response_A, **kwargs) # switched outputs

        original_pred = self.extract_label(original)
        followup_pred = self.extract_label(followup)

        # assign correctness points
        pred_correct = 0
        if (original_label == original_pred) and (followup_label == followup_pred):
            pred_correct = 1 # both are correct and positionally invariant
        elif (original_label == original_pred) or (followup_label == followup_pred):
            pred_correct = 0.5 # one was correct, but some positional bias was present

        # prepare output
        original = add_prefix_to_keys(original, "original_")
        followup = add_prefix_to_keys(followup, "followup_")
        original.update({
            **followup,
            "original_label": original_label,
            "followup_label": followup_label,
            "original_pred": original_pred, 
            "followup_pred": followup_pred,
            "pred_correct": pred_correct,
        })

        return original
    
    @staticmethod
    def invert_label(label):
        if label == ResponseQuality.A_BETTER:
            return ResponseQuality.B_BETTER
        elif label == ResponseQuality.B_BETTER:
            return ResponseQuality.A_BETTER
        return label # TIE stays the same

    def evaluate_batch(self, prompts, texts, return_mean=True):
        all_scores = []
        for prompt, text in zip(prompts, texts):
            chat = [{"role": "user", "content": prompt},{"role": "assistant", "content": text}]
            chat = self.tokenizer.apply_chat_template(chat, tokenize=True, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                score = self.model(chat).logits[0][0].item() 
                all_scores.append(score)  # Fixed typo from scor to score
        all_scores = np.array(all_scores)
        return all_scores.mean() if return_mean else all_scores
    


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

        oracle = NicolAIOracle()

        # Run quality assessments

        print("EVAL oracle.test:")
        start = time.time()
        results = oracle.test(instruction, original_text, mutated_text, label)
        delta = time.time() - start
        print(results)
        print("time_taken:", delta)
        

    test()
    