from abc import ABC, abstractmethod
from functools import partial

def apply_prompt(datarow, llm, prompt_fn, input_keys, output_keys, origin_A, origin_B, persona=None, prefix=""):
    inputs = {k: datarow[k] for k in input_keys}
    inputs['A'] = origin_A
    inputs['B'] = origin_B
    output = llm+prompt_fn(persona=persona, **inputs)
    return {prefix+k: output[k] for k in output_keys}

def flip_choices(output, prefix=""):
    output[f"{prefix}choice"] = "A" if output[f"{prefix}choice"] == "B" else "B"
    return output

# Abstract base class for all distinguishers
class Distinguisher(ABC):
    def __init__(self, llm, persona, origin_A, origin_B):
        self.llm = llm
        self.persona = persona
        self.origin_A = origin_A
        self.origin_B = origin_B

    @abstractmethod
    def prompt_fn(self, lm, persona, **kwargs):
        pass

    def distinguish(self, dataset, prefix=""):
        return dataset.map(
            partial(apply_prompt, 
                llm=self.llm, 
                prompt_fn=self.prompt_fn, 
                input_keys=self.input_keys, 
                output_keys=self.output_keys, 
                origin_A=self.origin_A,
                origin_B=self.origin_B,
                persona=self.persona,
                prefix=prefix
        )).map(
            partial(apply_prompt, 
                llm=self.llm, 
                prompt_fn=self.prompt_fn, 
                input_keys=self.input_keys, 
                output_keys=self.output_keys, 
                origin_A=self.origin_B,
                origin_B=self.origin_A,
                persona=self.persona,
                prefix=f"{prefix}flipped_"
        )).map(
            partial(flip_choices, prefix=f"{prefix}flipped_")
        )

    @property
    @abstractmethod
    def input_keys(self):
        pass

    @property
    @abstractmethod
    def output_keys(self):
        """
        Important: must include "choice" property that is either "A" or "B", representing the distinguisher's answer
        """
        pass