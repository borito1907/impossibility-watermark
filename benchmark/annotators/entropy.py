import guidance
from guidance import gen, select, user, system, assistant
from benchmark.annotators.annotator import Annotator

class EntropyMinimal(Annotator):
    @property
    def input_keys(self):
        return ["prompt"]
    
    @property
    def output_keys(self):
        return ["entropy_level", "entropy_analysis"]
    
    @staticmethod
    @guidance
    def annotation_fn(lm, persona, **kwargs):
        if persona:
            with system():
                lm += f"{persona}"
        with user():
            lm += f"""\
            ### Introduction
            I am trying to assess whether the given instructions have low, medium, or high entropy potential.
            
            Please assess the entropy of the instructions based on the following criteria:
                - How likely are the answers to use varied language and content?
                - Instructions that are unlikely to produce the same answer when asked again have high entropy.
            
            ### Task Description: 
            For the given instruction:
                - provide a one-sentence summary of the provided instruction.
                - provide a one-sentence analysis of the instruction entropy.
                - use your analysis to assign an entropy score from 0-9, where 0 is low and 9 is high.
            ### The instructions to evaluate:
            {kwargs['prompt']}
            """
        with assistant():
            lm += f"""\
            ### Entropy Assessment: 
            Instruction summary: {gen(f'instruction_summary', stop='.')}
            Here is my analysis of the entropy: {gen(f'entropy_analysis', stop='.')}
            Entropy Score: {gen(regex='[0-9]', name='entropy_level')}
            """
            # Entropy: {select(['low', 'medium', 'high'], name='entropy_level')}
        return lm

class EntropyBySummary(Annotator):
    @property
    def input_keys(self):
        return ["prompt"]
    
    @property
    def output_keys(self):
        return ["entropy_analysis", "entropy_level"]
    
    @staticmethod
    @guidance
    def annotation_fn(lm, persona, **kwargs):
        if persona:
            with system():
                lm += f"{persona}"
        with user():
            lm += f"""\
            ### Introduction
            In this context, entropy refers to the unpredictability and variety of the responses. 
            Instructions with high entropy will lead to responses that are less predictable and more varied, while low entropy instructions will be more predictable and uniform.
            
            ### Task Description: 
            1. Please assess the entropy of the instructions based on the following criteria:
                - Unpredictability Potential: How likely are the instructions to elicit a wide range of unpredictable responses?
                - Variety Potential: How likely are the instructions to generate responses with varied language and content?
                - Informativeness Potential: How likely are the responses to provide diverse and informative content beyond the basic requirements?
            - provide a one-sentence analysis of their potential entropy.
            2. For the given instructions:
                - provide a one-sentence summary of the instruction.
                - provide a one-sentence analysis of your reasoning for potential entropy.
                - assign a score from 0 to 9, where 0 indicates very low entropy potential and 9 indicates very high entropy potential.

            ### The instructions to evaluate:
            {kwargs['prompt']}
            """
        with assistant():
            lm += f"""\
            ### Entropy Assessment: 
            Instruction Summary: {gen(f'instruction_summary', stop='.')}
            Entropy Analysis: {gen(f'entropy_analysis', stop='.')}
            Entropy Score: {gen(regex='[0-9]', name='entropy_level')}
            """
        return lm

class EntropyByInstructions(Annotator):
    @property
    def input_keys(self):
        return ["prompt"]
    
    @property
    def output_keys(self):
        return ["entropy_level_prompt"]

    @staticmethod
    @guidance
    def annotation_fn(lm, persona, **kwargs):
        if persona:
            with system():
                lm += f"{persona}"
        with user():
            lm += f"""\
            ### Introduction
            In this context, entropy refers to the unpredictability and variety of the responses. 
            Instructions with high entropy will lead to responses that are less predictable and more varied, while low entropy instructions will be more predictable and uniform.
            
            ### Task Description: 
            1. Please assess the entropy of the instructions based on the following criteria:
                - Unpredictability Potential: How likely are the instructions to elicit a wide range of unpredictable responses?
                - Variety Potential: How likely are the instructions to generate responses with varied language and content?
                - Informativeness Potential: How likely are the responses to provide diverse and informative content beyond the basic requirements?
            2. For the given instructions, assign a score from 0 to 9, where 0 indicates very low entropy potential and 9 indicates very high entropy potential.

            ### The instructions to evaluate:
            {kwargs['prompt']}
            """
        with assistant():
            lm += f"""\
            ### Entropy Assessment: 
            Entropy Score: {gen(regex='[0-9]', name='entropy_level_prompt')}
            """
        return lm

class EntropyWithExp(Annotator):
    @property
    def input_keys(self):
        return ["prompt"]

    @property
    def output_keys(self):
        return ["entropy_level_prompt_w_exp", "entropy_analysis"]
    
    @staticmethod
    @guidance
    def annotation_fn(lm, persona, **kwargs):
        if persona:
            with system():
                lm += f"{persona}"
        with user():
            lm += f"""\
            ### Introduction
            In this context, entropy refers to the unpredictability and variety of the responses. 
            Instructions with high entropy will lead to responses that are less predictable and more varied, while low entropy instructions will be more predictable and uniform.
            
            ### Task Description: 
            1. Please assess the entropy of the instructions based on the following criteria:
                - Unpredictability Potential: How likely are the instructions to elicit a wide range of unpredictable responses?
                - Variety Potential: How likely are the instructions to generate responses with varied language and content?
                - Informativeness Potential: How likely are the responses to provide diverse and informative content beyond the basic requirements?
            2. For the given instructions:
                - provide a one-sentence analysis of their potential entropy.
                - assign a score from 0 to 9, where 0 indicates very low entropy potential and 9 indicates very high entropy potential.

            ### The instructions to evaluate:
            {kwargs['prompt']}
            """
        with assistant():
            lm += f"""\
            ### Entropy Assessment: 
            Entropy Analysis: {gen(f'entropy_analysis', stop='.')}.
            Entropy Score: {gen(regex='[0-9]', name='entropy_level_prompt_w_exp')}
            """
        return lm

class EntropyByInstructionsAndResponses(Annotator):
    @property
    def input_keys(self):
        return ["prompt", "response_a", "response_b"]

    @property
    def output_keys(self):
        return ["entropy_level_prompt_and_responses"]

    @staticmethod
    @guidance
    def annotation_fn(lm, persona, **kwargs):
        if persona:
            with system():
                lm += f"{persona}"
        with user():
            lm += f"""\
            ### Introduction
            In this context, entropy refers to the unpredictability and variety of the responses. 
            Instructions with high entropy will lead to responses that are less predictable and more varied, while low entropy instructions will be more predictable and uniform.
            
            ### Task Description: 
            1. Please assess the entropy of the instructions based on the following criteria:
                - Unpredictability Potential: How likely are the instructions to elicit a wide range of unpredictable responses?
                - Variety Potential: How likely are the instructions to generate responses with varied language and content?
                - Informativeness Potential: How likely are the responses to provide diverse and informative content beyond the basic requirements?
            2. For the given instructions:
                - assign a score from 0 to 9, where 0 indicates very low entropy potential and 9 indicates very high entropy potential.

            ### The instructions to evaluate:
            {kwargs['prompt']}

            ### Sample generation #1:
            {kwargs['response_a']}

            ### Sample generation #1:
            {kwargs['response_b']}
            """
        with assistant():
            lm += f"""\
            ### Entropy Assessment: 
            Entropy Score: {gen(regex='[0-9]', name='entropy_level_prompt_and_responses')}
            """
        return lm