from abc import ABC, abstractmethod
from functools import partial
from enum import Enum

class ResponseQuality(Enum):
    A_BETTER = 1
    B_BETTER = 2
    TIE = 3

# Abstract base class for all oracles
class Oracle(ABC):
    
    judge = None

    def __init__(self, llm, explain=False) -> None:
        self.llm = llm
        self.explain=explain

    @property
    @abstractmethod
    def input_keys(self):
        pass

    @property
    @abstractmethod
    def output_keys(self):
        pass

    @staticmethod
    @abstractmethod
    def annotation_fn(lm, explain=False, **kwargs):
        pass

    @staticmethod
    def apply_annotation(input_dict, llm, annotation_fn, input_keys, output_keys, persona=None, prefix=""):
        inputs = {k: input_dict[k] for k in input_keys}
        output = llm+annotation_fn(persona=persona, **inputs)
        return {prefix+k: output[k] for k in output_keys}

    def annotate(self, input_dict):
        return self.apply_annotation(
            input_dict=input_dict, 
            llm=self.llm, 
            annotation_fn=partial(self.annotation_fn, explain=self.explain),
            input_keys=self.input_keys, 
            output_keys=self.output_keys
        )

    def annotate_dataset(self, dataset, prefix=""):
        return dataset.map(
            partial(self.apply_annotation, 
                llm=self.llm, 
                annotation_fn=partial(self.annotation_fn, explain=self.explain),
                input_keys=self.input_keys, 
                output_keys=self.output_keys, 
                prefix=prefix
        )
    )

    @abstractmethod
    def evaluate(self, instruction, response_A, response_B, **kwargs):
        pass

    @abstractmethod
    def extract_label(self, evaluation):
        pass

    @abstractmethod
    def is_quality_preserved(self, instruction, original_text, mutated_text, **kwargs):
        pass

    @abstractmethod
    def test(self, instruction, response_A, response_B, label, **kwargs):
        pass

    @staticmethod
    def invert_label(label):
        if label == ResponseQuality.A_BETTER:
            return ResponseQuality.B_BETTER
        elif label == ResponseQuality.B_BETTER:
            return ResponseQuality.A_BETTER
        return label # TIE stays the same
    