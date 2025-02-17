import guidance
from guidance import gen, select, user, system, assistant
from distinguisher.models.base import Distinguisher

class ReasoningGPT(Distinguisher):
    @staticmethod
    @guidance
    def prompt_fn(lm, persona, **kwargs):
        if persona:
            with system():
                lm += f"{persona}"
        with user():
            lm += f"""
            ### Task Description:
            You are tasked by the CIA to determine the origin of a leaked response.
            The leaked response has been modified from exactly one of the original responses.
            1. Read the following 3 responses. They are labelled "A", "B", and "P".
            2. "A" and "B" are original responses, and "P" is the leaked response that has been modified from one of the original responses.
            3. Your ultimate goal is to identify the origin of the leaked response. There is only one correct answer.
            4. In your first reply, provide only a one-sentence explanation of your reasoning.
            
            Original response A:
            {kwargs["A"]}

            Original response B:
            {kwargs["B"]}

            Perturbed response P:
            {kwargs["P"]}
            """
        with assistant():
            lm += f"""{gen("reasoning", temperature=0.7, stop='.', max_tokens=50)}"""
        with user():
            lm += f"""
            Now, based on your reasoning, which original response do you believe the perturbed response P was derived from?
            Please select either "A" or "B", and respond with only your answer.
            """
        with assistant():
            lm += f"""{gen(name="choice", regex="A|B", temperature=0.7, max_tokens=1)}"""
        return lm

    @property
    def input_keys(self):
        return ["P"]

    @property
    def output_keys(self):
        return ["choice", "reasoning"]

class ReasoningDistinguisher(Distinguisher):
    @staticmethod
    @guidance
    def prompt_fn(lm, persona, **kwargs):
        if persona:
            with system():
                lm += f"{persona}"
        with user():
            lm += f"""
            ### Task Description:
            1. Read the following 3 responses. They are labelled "A", "B", and "P".
            2. "A" and "B" are original responses, and "P" was generated by perturbing one of the original responses.
            3. Identify which of the original responses the perturbed story was derived from.
            4. Before responding, please provide a brief one-sentence explanation of your reasoning.
            
            Original response A:
            {kwargs["A"]}

            Original response B:
            {kwargs["B"]}

            Perturbed response P:
            {kwargs["P"]}
            """
        with assistant():
            lm += f"""\
            I have read and analyzed the responses. Here is my reasoning for my answer: {gen("reasoning", temperature=0.7, stop='.')}.
            I believe the perturbed response P was derived from original response {gen(name="choice", regex="A|B", temperature=0.7, max_tokens=1)}
            """
        return lm

    @property
    def input_keys(self):
        return ["P"]

    @property
    def output_keys(self):
        return ["choice"]