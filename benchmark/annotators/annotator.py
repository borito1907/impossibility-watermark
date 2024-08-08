from abc import ABC, abstractmethod
from functools import partial

def apply_annotation(example, llm, annotation_fn, input_keys, output_keys, persona=None, prefix=""):
    inputs = {k: example[k] for k in input_keys}
    output = llm+annotation_fn(persona=persona, **inputs)
    return {prefix+k: output[k] for k in output_keys}

# Abstract base class for all annotators
class Annotator(ABC):
    def __init__(self, llm, persona):
        self.llm = llm
        self.persona = persona

    @staticmethod
    @abstractmethod
    def annotation_fn(lm, persona, **kwargs):
        pass

    def annotate(self, dataset, prefix=""):
        return dataset.map(
            partial(apply_annotation, 
                llm=self.llm, 
                annotation_fn=self.annotation_fn, 
                input_keys=self.input_keys, 
                output_keys=self.output_keys, 
                prefix=prefix
        )
    )

    @property
    @abstractmethod
    def input_keys(self):
        pass

    @property
    @abstractmethod
    def output_keys(self):
        pass