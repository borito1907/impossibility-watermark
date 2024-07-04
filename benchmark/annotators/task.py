import guidance
from guidance import gen, select, user, system, assistant
from benchmark.annotators.annotator import Annotator

n = '\n'

class FreeTask(Annotator):
    @property
    def input_keys(self):
        return ["prompt"]

    @property
    def output_keys(self):
        return ["task"]

    @staticmethod
    @guidance
    def annotation_fn(lm, persona, **kwargs):
        if persona:
            with system():
                lm += f"{persona}"
        with user():
            lm += f"""
            ### Introduction: 
            Your task is to read each instruction given to a language model and identify the specific task it represents. 
            This involves understanding the intent of the instruction and categorizing it accordingly.
            Tasks could include summarization, translation, data retrieval, creative writing, etc.

            ### Task Description: 
            1. Read the Instruction: Carefully read the instruction provided to the language model.
            2. Identify the Task: Determine the main task that the instruction is asking the language model to perform. 
            3. If the task does not fit any category perfectly, use 'other'.
            
            ### The prompt to evaluate:
            {kwargs["prompt"]}
            """
        with assistant():
            lm += f"""\
            Most relevant task: {gen(stop=n, max_tokens=5, name='task')}
            """
        return lm

class TaskMinimalist(Annotator):
    @property
    def input_keys(self):
        return ["prompt"]
    
    @property
    def output_keys(self):
        return ["best_task"]

    @staticmethod
    @guidance
    def annotation_fn(lm, persona, **kwargs):
        if persona:
            with system():
                lm += f"{persona}"
        with user():
            lm += f"""\
            ### Introduction
            You must annotate a prompt as relevant to a particular task. Here are the task options and their definitions:
            
            Tasks:
                - Generation : prompt that ask you for creative writing
                - Brainstorming : prompt that ask you to generate ideas
                - Classification : prompt that ask you to label or categorize items
                - Extraction : prompt that ask you to identify certain words present in the instruction
                - Summarization : prompt that ask you to summarize information present in the instruction
                - Rewriting : prompt that ask you to rewrite information present in the instruction
                - Chat : prompt that ask you to engage in a conversation
                - Open QA : prompt that ask you to answer a question based on your knowledge
                - Closed QA : prompt that ask you to answer a question based on knowledge provided in the instruction

            ### Task Description: 
            1. Carefully read the given prompt and determine the most appropriate task. 
            2. If no task is appropriate, select "Other".

            ### The prompt to evaluate:
            {kwargs["prompt"]}
            """
        with assistant():
            lm += f"""\
            Most relevant task: {select(['Generation', 'Brainstorming', 'Classification', 'Extraction', 'Summarization', 'Rewriting', 'Chat', 'Open QA', 'Closed QA', 'Other'], name='best_task')}
            """
        return lm

class TaskMinimal(Annotator):
    @property
    def input_keys(self):
        return ["prompt"]
    
    @property
    def output_keys(self):
        return ["best_task", "task_explanation"]

    @staticmethod
    @guidance
    def annotation_fn(lm, persona, **kwargs):
        if persona:
            with system():
                lm += f"{persona}"
        with user():
            lm += f"""\
            ### Introduction
            I am trying to identify the type of task that a given instruction represents in order to study the differences in the structure of responses.
            You will be grouping the provided instructions into a category based on the type of task they represent.
            
            ### Task Description: 
            1. Please use the following category definitions in your assessment:
                - Generation: instructions that ask you for creative writing
                - Brainstorming: instructions that ask you to generate ideas
                - Classification: instructions that ask you to label or categorize items
                - Extraction: instructions that ask you to identify certain words present in the instruction
                - Summarization: instructions that ask you to summarize information present in the instruction
                - Rewriting: instructions that ask you to rewrite information present in the instruction
                - Chat: instructions that ask you to engage in a conversation
                - Open QA: instructions that ask you to answer a question based on your knowledge
                - Closed QA: instructions that ask you to answer a question based on knowledge provided in the instruction

            2. For the given instructions:
                - provide a one-sentence summary of the provided instruction.
                - provide a one-sentence explanation for the best category.
                - select the best category that fits the instruction.

            ### The instructions to evaluate:
            {kwargs["prompt"]}
            """
        with assistant():
            lm += f"""\
            Instruction summary: {gen(f'instruction_summary', stop='.')}
            Category explanation: {gen(f'task_explanation', stop='.')}
            Best Category: {select(['Generation', 'Brainstorming', 'Classification', 'Extraction', 'Summarization', 'Rewriting', 'Chat', 'Open QA', 'Closed QA'], name='best_task')}
            """
        return lm

class TaskBySummary(Annotator):
    @property
    def input_keys(self):
        return ["prompt"]
    
    @property
    def output_keys(self):
        # return ["best_task", "task_explanation"]
        return ["best_task", "generation_score", "brainstorming_score", "classification_score", "extraction_score", "summarization_score", "rewriting_score", "chat_score", "open_qa_score", "closed_qa_score", "instruction_summary", "task_explanation"]

    @staticmethod
    @guidance
    def annotation_fn(lm, persona, **kwargs):
        if persona:
            with system():
                lm += f"{persona}"
        with user():
            lm += f"""\
            ### Introduction
            You will be assessing the task category of the provided instructions. 
            Task categories group instructions based on their use case.
            
            ### Task Description: 
            1. Please use the following category definitions in your assessment:
                - Generation: instructions that ask you for creative writing
                - Brainstorming: instructions that ask you to generate ideas
                - Classification: instructions that ask you to label or categorize items
                - Extraction: instructions that ask you to identify certain words present in the instruction
                - Summarization: instructions that ask you to summarize information present in the instruction
                - Rewriting: instructions that ask you to rewrite information present in the instruction
                - Chat: instructions that ask you to engage in a conversation
                - Open QA: instructions that ask you to answer a question based on your knowledge
                - Closed QA: instructions that ask you to answer a question based on knowledge provided in the instruction

            2. For the given instructions:
                - provide a one-sentence summary of what the instruction is asking you to do.
                - provide a one-sentence explanation for the best category.
                - assign a score from 0 to 9 to each category, where 0 indicates very low relevance to the category and 9 indicates very high relevance to the category.

            ### The instructions to evaluate:
            {kwargs["prompt"]}
            """
        with assistant():
            lm += f"""\
            Instruction summary: {gen(f'instruction_summary', stop='.')}
            Category explanation: {gen(f'task_explanation', stop='.')}
            Category Scores:
            - Generation: {gen(regex='[0-9]', name='generation_score')}
            - Brainstorming: {gen(regex='[0-9]', name='brainstorming_score')}
            - Classification: {gen(regex='[0-9]', name='classification_score')}
            - Extraction: {gen(regex='[0-9]', name='extraction_score')}
            - Summarization: {gen(regex='[0-9]', name='summarization_score')}
            - Rewriting: {gen(regex='[0-9]', name='rewriting_score')}
            - Chat: {gen(regex='[0-9]', name='chat_score')}
            - Open QA: {gen(regex='[0-9]', name='open_qa_score')}
            - Closed QA: {gen(regex='[0-9]', name='closed_qa_score')}
            Best Category: {select(['Generation', 'Brainstorming', 'Classification', 'Extraction', 'Summarization', 'Rewriting', 'Chat', 'Open QA', 'Closed QA'], name='best_task')}
            """
        return lm

class TaskByScores(Annotator):
    @property
    def input_keys(self):
        return ['prompt']

    @property
    def output_keys(self):
        return ["best_task", "generation_score", "brainstorming_score", "classification_score", "extraction_score", "summarization_score", "rewriting_score", "chat_score", "open_qa_score", "closed_qa_score", "other_score", "top_three_categories", "task_explanation"]

    @staticmethod
    @guidance
    def annotation_fn(lm, persona, **kwargs):
        if persona:
            with system():
                lm += f"{persona}"
        with user():
            lm += f"""\
            ### Introduction
            You will be assessing the task category of the provided instructions. 
            Task categories group instructions based on their use case.
            For example, instructions for writing a poem would fall under 'Generation', while instructions 
            
            ### Task Description: 
            1. Please categorize the instructions into one of the following categories
                - Generation: instructions that require creative output such as writing an essay
                - Brainstorming: instructions that require coming up with multiple ideas such as listing baby names
                - Classification: instructions that require labeling or categorizing items such as rating sarcasm
                - Extraction: instructions that require extracting information such as identifying countries in a text
                - Summarization: instructions that require summarizing provided information
                - Rewriting: instructions that require rewriting information in a different context
                - Chat: instructions that require engaging in a conversation
                - Open QA: instructions that require answering questions based on general knowledge
                - Closed QA: instructions that require answering questions based on specific information
                - Other: instructions that do not fit into the above categories

            2. For the given instructions:
                - assign a score from 0 to 9 to each category, where 0 indicates very low relevance to the category and 9 indicates very high relevance to the category.
                - list the top three categories that best fit the instructions.
                - provide a one-sentence explanation for the best category.
                - select the category that best fits the instructions.

            ### The instructions to evaluate:
            {kwargs['prompt']}
            """
        with assistant():
            lm += f"""\
            ### Task Category Scores: 
            - Generation: {gen(regex='[0-9]', name='generation_score')}
            - Brainstorming: {gen(regex='[0-9]', name='brainstorming_score')}
            - Classification: {gen(regex='[0-9]', name='classification_score')}
            - Extraction: {gen(regex='[0-9]', name='extraction_score')}
            - Summarization: {gen(regex='[0-9]', name='summarization_score')}
            - Rewriting: {gen(regex='[0-9]', name='rewriting_score')}
            - Chat: {gen(regex='[0-9]', name='chat_score')}
            - Open QA: {gen(regex='[0-9]', name='open_qa_score')}
            - Closed QA: {gen(regex='[0-9]', name='closed_qa_score')}
            - Other: {gen(regex='[0-9]', name='other_score')}

            ### Category Assessment: 
            Top three categories: {gen(f'top_three_categories', stop='.')}
            Explanation: {gen(f'task_explanation', stop='.')}
            Best Category: {select(['Generation', 'Brainstorming', 'Classification', 'Extraction', 'Summarization', 'Rewriting', 'Chat', 'Open QA', 'Closed QA', 'Other'], name='best_task')}
            """
        return lm