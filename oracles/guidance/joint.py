# RUN: CUDA_VISIBLE_DEVICES=4,5,6,7 python -m oracles.guidance.rank

import guidance
from guidance import gen, select, user, system, assistant
from oracles.base import Oracle, ResponseQuality
from oracles.utils import add_prefix_to_keys

class JointOracle(Oracle):

    @property
    def input_keys(self):
        return ["instruction", "response_A", "response_B",]

    @property
    def output_keys(self):
        return ["assistant_1_score", "assistant_2_score", "explanation"] if self.explain else ["assistant_1_score", "assistant_2_score"]

    @staticmethod
    @guidance
    def annotation_fn(lm, explain=False, **kwargs):
        pattern = '[1-9]|10'
        newline = "\n"
        if kwargs["persona"] is not None:
            with system():
                lm += f"{kwargs['persona']}"
        with user():
            lm += f"""\
            ### Instructions: 

            We would like to request your feedback on the performance of two AI assistants in response to the user prompt displayed below.
            Please rate the grammatical correctness, fluency, accuracy, consistency, and clarity. 
            Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.

            ### Here is the prompt:
            {kwargs['instruction']}

            ### Assistant 1 Response:
            {kwargs['response_A']}

            ### Assistant 2 Response:
            {kwargs['response_B']}
            """
        with assistant():
            if explain:
                lm += f"""\
                ### Explanation:
                {gen(name='explanation', max_tokens=200, stop=["<|eot_id|>", newline])}
                """
            lm += f"""\
            ### Feedback: 
            Assistant 1 Score: {gen(name='assistant_1_score', regex=pattern, max_tokens=2)}
            Assistant 2 Score: {gen(name='assistant_2_score', regex=pattern, max_tokens=2)}
            """
        return lm

    def evaluate(self, instruction, response_A, response_B, **kwargs):
        input_dict = {
            "instruction": instruction, 
            "response_A": response_A,
            "response_B": response_B
        }
        evaluation = self.annotate(input_dict)
        return evaluation

    def extract_label(self, evaluation):
        assistant_1_score = int(evaluation["assistant_1_score"])
        assistant_2_score = int(evaluation["assistant_2_score"])
        if assistant_1_score > assistant_2_score:
            label = ResponseQuality.A_BETTER
        elif assistant_1_score < assistant_2_score:
            label = ResponseQuality.B_BETTER
        else:
            label = ResponseQuality.TIE
        return label

    def is_quality_preserved(self, instruction, original_text, mutated_text, **kwargs):
        
        original = self.evaluate(instruction, response_A=original_text, response_B=mutated_text, **kwargs) 
        followup = self.evaluate(instruction, response_A=mutated_text, response_B=original_text, **kwargs) # switched outputs
        
        original_pred = self.extract_label(original)
        followup_pred = self.extract_label(followup)
        
        if original_pred in [ResponseQuality.B_BETTER, ResponseQuality.TIE] and followup_pred in [ResponseQuality.A_BETTER, ResponseQuality.TIE]:
            is_quality_preserved = True
        else:
            is_quality_preserved = False

        original = add_prefix_to_keys(original, "original_")
        followup = add_prefix_to_keys(followup, "followup_")
        original.update({**followup})
        original.update({"quality_preserved": is_quality_preserved})
        return original

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

# Testing
if __name__ == "__main__":

    import pandas as pd
    import time
    import warnings

    warnings.filterwarnings("error")

    def test():

        from guidance import models        

        # Load sample data row
        dataset = pd.read_csv("./data/lmsys-14x100-grouped.csv")
        dataset = dataset.sample(frac=1).reset_index(drop=True)
        dataset = dataset[dataset["winner_tie"] == 0].head(1) 
        instruction = dataset["prompt"][0]
        original_text = dataset["response_a"][0]
        mutated_text = dataset["response_b"][0]
        label = ResponseQuality.TIE if dataset["winner_tie"][0] else ResponseQuality.A_BETTER if dataset["winner_model_a"][0] else ResponseQuality.B_BETTER

        # Initialize Base LLM
        print("Initializing Base LLM with Meta-Llama-3-8B-Instruct/ggml-model-q8_0.gguf")
        model_id = "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct/ggml-model-q8_0.gguf"
        llm = models.LlamaCpp(
            model=model_id,
            echo=False,
            n_gpu_layers=-1,
            n_ctx=2048
        )

        oracle = JointOracle(llm, explain=False)

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

        # # Initialize Base LLM - FHC: THIS DOES NOT WORK RELIABLY RIGHT NOW. 
        # print("Initializing Oracle with gpt-4o-mini...")
        # model_id = "gpt-4o-mini"
        # llm = models.OpenAI(
        #     model=model_id
        # )

        # oracle = JointOracle(llm, explain=False)

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
        
        # print("EVAL  oracle.test:")
        # start = time.time()
        # results = oracle.test(instruction, original_text, mutated_text, label)
        # delta = time.time() - start
        # print(results)
        

    test()
    