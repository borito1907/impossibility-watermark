from abc import ABC, abstractmethod
from functools import partial
import datasets

def apply_prompt(datarow, llm, prompt_fn, input_keys, output_keys, origin_A, origin_B, persona=None, prefix=""):
    inputs = {k: datarow[k] for k in input_keys}
    inputs['A'] = origin_A
    inputs['B'] = origin_B
    output = llm+prompt_fn(persona=persona, **inputs)
    return {prefix+k: output[k] for k in output_keys}

def flip_choices(output, prefix=""):
    if output[f"{prefix}choice"] == "A":
        output[f"{prefix}choice"] = "B"
    elif output[f"{prefix}choice"] == "B":
        output[f"{prefix}choice"] = "A"
    else:
        # sad face model didnt work
        pass
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
    
    def distinguish_majority(self, dataset, count, prefix=""):
        print(f"Running distinguisher on {dataset.num_rows} rows with a count of {count} for a total of {dataset.num_rows*count} trials")
        for i in range(count):
            dataset = self.distinguish(dataset, prefix=f"{prefix}{i}_")
        df = dataset.to_pandas()
        df[f"{prefix}A_normal"] = df[[f"{prefix}{i}_choice" for i in range(count)]].apply(lambda row: (row == 'A').sum(), axis=1)
        df[f"{prefix}B_normal"] = df[[f"{prefix}{i}_choice" for i in range(count)]].apply(lambda row: (row == 'B').sum(), axis=1)
        df[f"{prefix}A_flipped"] = df[[f"{prefix}{i}_flipped_choice" for i in range(count)]].apply(lambda row: (row == 'A').sum(), axis=1)
        df[f"{prefix}B_flipped"] = df[[f"{prefix}{i}_flipped_choice" for i in range(count)]].apply(lambda row: (row == 'B').sum(), axis=1)

        # drop the individual columns
        df = df.drop(columns=[f"{prefix}{i}_choice" for i in range(count)])
        df = df.drop(columns=[f"{prefix}{i}_flipped_choice" for i in range(count)])
        
        df[f"{prefix}A_count"] = df[f"{prefix}A_normal"] + df[f"{prefix}A_flipped"]
        df[f"{prefix}B_count"] = df[f"{prefix}B_normal"] + df[f"{prefix}B_flipped"]
        df[f"{prefix}choice"] = df.apply(lambda x: "A" if x[f"{prefix}A_count"] > x[f"{prefix}B_count"] else "B", axis=1)
        df[f"{prefix}flipped_choice"] = df.apply(lambda x: "A" if x[f"{prefix}A_count"] >= x[f"{prefix}B_count"] else "B", axis=1)

        # normalize the counts
        df[f"{prefix}A_normal"] /= count
        df[f"{prefix}B_normal"] /= count
        df[f"{prefix}A_flipped"] /= count
        df[f"{prefix}B_flipped"] /= count
        df[f"{prefix}A_count"] /= count*2
        df[f"{prefix}B_count"] /= count*2

        return datasets.Dataset.from_pandas(df)

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