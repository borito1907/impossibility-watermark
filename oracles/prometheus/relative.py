import logging
import warnings
from dotenv import load_dotenv

from oracles.base import ResponseQuality
from prometheus_eval.vllm import VLLM
from prometheus_eval.litellm import AsyncLiteLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import RELATIVE_PROMPT, ABSOLUTE_PROMPT

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class PrometheusRelativeOracle:
    """Relative Grading: Outputs A or B"""
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
        self.rubric = "Determine which response best addresses the instructions. Your evaluation should consider factors such as the repetition, grammar, coherence, relevance, accuracy of the responses. Note that having grammatical errors, repetitions, capitalization errors or punctuation mistakes would greatly degrade the quality of a response."

        # Loading model
        if "8x7b" in self.model_id and self.num_gpus < 4:
            self.num_gpus = 4
            warnings.warn(
                f"`prometheus-8x7b-v2.0` requires ~172GB of GPU RAM. Increasing num_gpus from {self.num_gpus} to 4."
            )
        self.load_judge()
    
    def load_judge(self):
        # Load or initialize the model used for scoring and feedback
        if Oracle.judge == None:
            if "gpt-" in self.model_id:
                # Assume OpenAI (see https://github.com/prometheus-eval/prometheus-eval/tree/main?tab=readme-ov-file#llm-apis)
                load_dotenv()
                self.model = AsyncLiteLLM(self.model_id)
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


    def evaluate(self, instruction, response_A, response_B, reference_answer=None):
        feedback, score = self.judge.single_relative_grade(
            rubric=self.rubric,
            instruction=instruction,
            response_A=response_A,
            response_B=response_B,
            reference_answer=reference_answer
        )
        print(f"feedback: {feedback}")
        print(f"score: {score}")
        return feedback, score

    def evaluate_batch(self, instructions, responses_A, responses_B, reference_answers=None):
        feedbacks, scores = self.judge.relative_grade(
            rubric=self.rubric,
            instructions=instructions,
            responses_A=responses_A,
            responses_B=responses_B,
            reference_answers=reference_answers,
            params={},
        )
        print(f"feedbacks: {feedbacks}")
        print(f"scores: {scores}")
        return feedbacks, scores

    def is_quality_preserved(self, instruction, original_text, mutated_text, reference_answer=None):
        # Prepare evaluation
        print(self.model_id)
        if "gpt-" in self.model_id:
            feedback_1, score_1 = self.evaluate(instruction, original_text, mutated_text)
            feedback_2, score_2 = self.evaluate(instruction, mutated_text, original_text)
            feedbacks = [feedback_1, feedback_2]
            scores = [score_1, score_2]
        else:
            instructions = [instruction] * 2
            responses_A = [original_text, mutated_text]
            responses_B = [mutated_text, original_text]
            reference_answers = [reference_answer] * 2 if reference_answer else [None] * 2
            feedbacks, scores = self.evaluate_batch(instructions, responses_A, responses_B, reference_answers)

        # Determine if quality is preserved by comparing flipped scores
        # NOTE: We resolve positional bias in favor of quality being preserved.
        if "A" in scores[0] and "B" in scores[1]:
            quality_preserved = False
        else:
            quality_preserved = True

        # Package results for output
        quality_eval =  {
            "feedbacks": feedbacks,
            "scores": scores,
            "quality_preserved": quality_preserved
        }

        return quality_eval
    
    def extract_label(self, scores):
        if "A" in scores[0] and "B" in scores[1]:
            return ResponseQuality.A_BETTER
        elif "B" in scores[0] and "A" in scores[1]:
            return ResponseQuality.B_BETTER
        else:
            return ResponseQuality.TIE
    
    def test(self, instruction, response_A, response_B, label, **kwargs):
        # Prepare evaluation
        if "gpt-" in self.model_id:
            feedback_1, score_1 = self.evaluate(instruction, response_A, response_B)
            feedback_2, score_2 = self.evaluate(instruction, response_B, response_A)
            feedbacks = [feedback_1, feedback_2]
            scores = [score_1, score_2]
        else:
            instructions = [instruction] * 2
            responses_A = [response_A, response_B]
            responses_B = [response_B, response_A]
            reference_answers = [None] * 2
            feedbacks, scores = self.evaluate_batch(instructions, responses_A, responses_B, reference_answers)
        
        pred = self.extract_label(scores)
        
		# assign correctness points
        pred_correct = 0
        if (label == pred):
            pred_correct = 1 
            
        results = {
            "response_A_feedback": feedbacks[0],
            "response_A_score": scores[0],  
            "response_B_feedback": feedbacks[1],
            "response_B_score": scores[1],  
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
        oracle = PrometheusRelativeOracle(
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

        print("Test Prometheus Relative Oracle:")
        start = time.time()
        results = oracle.test(instruction, original_text, mutated_text, label)
        delta = time.time() - start
        print(results)
        print("time_taken:", delta)

        # feedback, score = oracle.evaluate(instruction, mutated_text, original_text)
        # print("Evaluation WITH Reference Answer")
        # print("Feedback:", feedback)
        # print("Score:", score)

        # print("Evaluation WITHOUT Reference Answer")
        # feedback, score = oracle.evaluate(instruction, mutated_text, None)
        # print("Feedback:", feedback)
        # print("Score:", score)
        # Initialize Oracle

        # # NOTE: this is currently unreliable for relative oracle... added github issue: https://github.com/prometheus-eval/prometheus-eval/issues/46
        # print("Initializing Prometheus with gpt-4o...")
        # oracle = PrometheusRelativeOracle(
        #     model_id="gpt-4o"
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
        
        # print("Test Prometheus Relative Oracle:")
        # start = time.time()
        # results = oracle.test(instruction, original_text, mutated_text, label)
        # delta = time.time() - start
        # print(results)
        
    test()
    