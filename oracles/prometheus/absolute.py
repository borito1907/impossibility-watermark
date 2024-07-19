# Absolute Grading: Outputs score of 1 to 5
import logging
import warnings
from dotenv import load_dotenv

from oracles.base import ResponseQuality
from prometheus_eval.vllm import VLLM
from prometheus_eval.litellm import AsyncLiteLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE, RELATIVE_PROMPT

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class PrometheusAbsoluteOracle:
    def __init__(
        self, 
        model_id="prometheus-eval/prometheus-8x7b-v2.0",
        download_dir="/data2/.shared_models",
        num_gpus=4, 
    ):
        # Initialize any necessary attributes or models here
        self.model_id = model_id
        self.download_dir = download_dir
        self.num_gpus = num_gpus

        # Defining rubric criteria
        self.rubric_data = {
            "criteria": "Determine which response best addresses the instructions. Your evaluation should consider factors such as repetition, grammar, coherence, relevance, and accuracy of the responses. Note that having grammatical errors, repetitions, capitalization errors, or punctuation mistakes would greatly degrade the quality of a response.",
            "score1_description": "The response fails to address the instructions, with significant issues in grammar, coherence, relevance, and accuracy. The response contains multiple grammatical errors, repetitions, and capitalization or punctuation mistakes, greatly degrading its quality.",
            "score2_description": "The response partially addresses the instructions but has notable issues in grammar, coherence, relevance, and accuracy. There are several grammatical errors, repetitions, and capitalization or punctuation mistakes that detract from its quality.",
            "score3_description": "The response generally addresses the instructions with moderate issues in grammar, coherence, relevance, and accuracy. Some grammatical errors, repetitions, and capitalization or punctuation mistakes are present but do not severely impact the overall quality.",
            "score4_description": "The response effectively addresses the instructions with minor issues in grammar, coherence, relevance, and accuracy. Few grammatical errors, repetitions, and capitalization or punctuation mistakes are present and do not significantly affect the quality.",
            "score5_description": "The response excellently addresses the instructions with high standards of grammar, coherence, relevance, and accuracy. It is free from grammatical errors, repetitions, and capitalization or punctuation mistakes, demonstrating superior quality."
            }
        self.rubric = SCORE_RUBRIC_TEMPLATE.format(**self.rubric_data)

        # Loading model
        if "8x7b" in self.model_id and self.num_gpus < 4:
            self.num_gpus = 4
            warnings.warn(
                f"`prometheus-8x7b-v2.0` requires ~172GB of GPU RAM. Increasing num_gpus from {self.num_gpus} to 4."
            )
        self.load_judge()
    
    def load_judge(self):
        # Load or initialize the model used for scoring and feedback
        if "gpt-" in self.model_id:
            # Assume OpenAI (see https://github.com/prometheus-eval/prometheus-eval/tree/main?tab=readme-ov-file#llm-apis)
            load_dotenv()
            self.model = AsyncLiteLLM(self.model_id, requests_per_minute=100)
        else:
            # Assume Transformers
            self.model = VLLM(
                model=self.model_id, 
                tensor_parallel_size=self.num_gpus, 
                download_dir=self.download_dir
            )
        self.judge = PrometheusEval(
            model=self.model,
            relative_grade_template=RELATIVE_PROMPT,
            absolute_grade_template=ABSOLUTE_PROMPT					
        )

    def evaluate(self, instruction, response, reference_answer=None):
        feedback, score = self.judge.single_absolute_grade(
            rubric=self.rubric,
            instruction=instruction,
            response=response,
            reference_answer=reference_answer
        )
        return feedback, score

    def evaluate_batch(self, instructions, responses, reference_answers=None):
        feedbacks, scores = self.judge.absolute_grade(
            rubric=self.rubric,
            instructions=instructions,
            responses=responses,
            reference_answers=reference_answers,
            params={},
        )
        return feedbacks, scores

    def is_quality_preserved(self, instruction, original_text, mutated_text, reference_answer=None):
        # Prepare evaluation
        instructions = [instruction] * 2
        responses = [original_text, mutated_text]
        reference_answers = [reference_answer] * 2 if reference_answer else [None] * 2
        feedbacks, scores = self.evaluate_batch(instructions, responses, reference_answers)

        # Determine if quality is preserved by comparing scores
        if scores[1] >= scores[0]:
            quality_preserved = True
        else:
            quality_preserved = False

        # Package results for output
        quality_eval =  {
            "feedbacks": feedbacks,
            "scores": scores,
            "quality_preserved": quality_preserved
        }

        return quality_eval
    
    def extract_label(self, evaluation):
        return evaluation[1]

    def derive_label(self, score1, score2):
        if score1 > score2:
            label = ResponseQuality.A_BETTER
        elif score1 < score2:
            label = ResponseQuality.B_BETTER
        else:
            label = ResponseQuality.TIE
        return label
    
    def test(self, instruction, response_A, response_B, label, **kwargs):
        response_A_evaluation = self.evaluate(instruction, response_A)
        response_B_evaluation = self.evaluate(instruction, response_B)
        
        response_A_score = self.extract_label(response_A_evaluation)
        response_B_score = self.extract_label(response_B_evaluation)

        pred = self.derive_label(response_A_score, response_B_score)
        
		# assign correctness points
        pred_correct = 0
        if (label == pred):
            pred_correct = 1 
            
        results = {
            "response_A_feedback": response_A_evaluation[0],
            "response_A_score": response_A_score,  
            "response_B_feedback": response_B_evaluation[0],
            "response_B_score": response_B_score,  
            "original_label": label,
            "followup_label": "NA",
            "original_pred": pred, 
            "followup_pred": "NA",
            "pred_correct": pred_correct,
		}
				
        return results
        

# Testing
if __name__ == "__main__":

    import pandas as pd
    import time

    def test():
        
        # Load sample data row
        dataset = pd.read_csv("./data/lmsys-14x100-grouped.csv")
        dataset = dataset.sample(frac=1).reset_index(drop=True)
        dataset = dataset[dataset["winner_tie"] == 0].head(1) 
        instruction = dataset["prompt"].iloc[0]
        original_text = dataset["response_a"].iloc[0]
        mutated_text = dataset["response_b"].iloc[0]
        label = ResponseQuality.TIE if dataset["winner_tie"].iloc[0] else ResponseQuality.A_BETTER if dataset["winner_model_a"].iloc[0] else ResponseQuality.B_BETTER

        # Initialize Oracle
        print("Initializing Prometheus with prometheus-8x7b-v2.0...")
        oracle = PrometheusAbsoluteOracle(
            model_id="prometheus-eval/prometheus-8x7b-v2.0",
            download_dir="/data2/.shared_models",
            num_gpus=4
        )

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

        # print("Test Prometheus Absolute Oracle:")
        # start = time.time()
        # results = oracle.test(instruction, original_text, mutated_text, label)
        # delta = time.time() - start
        # print(results)
        # print("time_taken:", delta)

        # feedback, score = oracle.evaluate(instruction, mutated_text, original_text)
        # print("Evaluation WITH Reference Answer")
        # print("Feedback:", feedback)
        # print("Score:", score)

        # print("Evaluation WITHOUT Reference Answer")
        # feedback, score = oracle.evaluate(instruction, mutated_text, None)
        # print("Feedback:", feedback)
        # print("Score:", score)
        # Initialize Oracle

        # print("Initializing Prometheus with gpt-4-turbo...")
        # oracle = PrometheusAbsoluteOracle(
        #     model_id="gpt-4-turbo"
        # )

        # # Run quality assessments
        # start = time.time()
        # quality_eval = oracle.is_quality_preserved(
        #     instruction=instruction, 
        #     original_text=original_text, 
        #     mutated_text=mutated_text, 
        #     reference_answer=None
        # )
        # delta = time.time() - start
        # print("EVAL oracle.is_quality_preserved")
        # print("quality_eval:", quality_eval)
        # print("time_taken:", delta)
        
        # print("Test Prometheus Absolute Oracle:")
        # start = time.time()
        # results = oracle.test(instruction, original_text, mutated_text, label)
        # delta = time.time() - start
        # print(results)
        
    test()
    