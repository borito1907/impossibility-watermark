# RUN: python -m oracles.training.openai.formatters

from abc import ABC, abstractmethod
import textwrap
import pandas as pd

class BaseFormatter(ABC):
    # The TEMPLATE should be defined in the subclasses
    TEMPLATE: str = ""
    system_message = """You are an expert in document evaluation. Compare the original and edited versions of a document to determine if the edit improved its quality. Assess based on clarity, coherence, grammar, style, content accuracy, and engagement."""


    def format(self, row: dict) -> dict:
        # Use the format method of strings to replace variables in the TEMPLATE
        content = self.TEMPLATE.format(**row)
        return {"content": content}

    def apply_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        # Apply the format method to each row in the DataFrame
        df['formatted_content'] = df.apply(lambda row: self.format(row.to_dict()), axis=1)
        return df


################
# BinaryOracle #
################

class BinaryOracleFormatter(BaseFormatter):
    TEMPLATE = textwrap.dedent("""
        ### Instructions: 
        We would like to request your feedback on the performance of two AI assistants in response to the user prompt displayed below.
        The second assistant's goal is to produce a response that is at least as good as the first assistant's response.
        Please rate whether the goal has been achieved based on the grammatical correctness, fluency, accuracy, consistency, and clarity.
        Be strict in your evaluation and consider the overall quality of the response.
        If the second assistant's response is at least as good, output "Yes", otherwise output "No".

        ### Here is the prompt:
        {prompt}

        ### Assistant 1 Response:
        {original_response}

        ### Assistant 2 Response:
        {mutated_response}
    """)

    def format(self, row: dict) -> dict:

        # user_content
        user_content = self.TEMPLATE.format(**row)

        # asst_content
        asst_content = yesno(row)

        # format output
        system_message = {"role": "system", "content": self.system_message}
        user_message   = {"role": "user", "content": user_content}
        asst_message   = {"role": "assistant", "content": asst_content}
        messages       = [system_message, user_message, asst_message]

        return {"messages": messages}

##############
# DiffOracle #
##############

import difflib
import regex as re 
from itertools import chain, zip_longest

def find_revisions(original, mutation):
    def intersperse_lists(list1, list2):
        return ''.join(chain(*zip_longest(list1, list2)))
        
    line1 = re.split(r'(\S+)', original)[1::2]
    line2 = re.split(r'(\S+)', mutation)[1::2]
    text1 = re.split(r'(\S+)', original)
    line1 = text1[1::2]
    whitespace1 = text1[::2]
    text2 = re.split(r'(\S+)', mutation)
    line2 = text2[1::2]
    whitespace2 = text2[::2]
    
    report = ""
    matcher = difflib.SequenceMatcher(None, line1, line2)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            report += intersperse_lists(whitespace1[i1:i2], line1[i1:i2]) 
        elif tag == 'delete':
            whitespace1[i1] = ''
            text = intersperse_lists(whitespace1[i1:i2], line1[i1:i2])
            report += f" <DELETE: \"{text}\">"
        elif tag == 'insert':
            whitespace2[j1] = ''
            text = intersperse_lists(whitespace2[j1:j2], line2[j1:j2])
            report += f" <INSERT: \"{text}\">"
        elif tag == 'replace':
            whitespace1[i1] = ''
            whitespace2[j1] = ''
            text1 = intersperse_lists(whitespace1[i1:i2], line1[i1:i2])
            text2 = intersperse_lists(whitespace2[j1:j2], line2[j1:j2])
            report += f" <REPLACE: \"{text1}\" WITH: \"{text2}\">"
    return report

def yesno(row):
    if "original" in row["selected"]:
        return "No"
    if "mutated" in row["selected"]:
        return "Yes"
    if "tie" in row["selected"]:
        return "Yes"

class DiffOracleFormatter(BaseFormatter):
    TEMPLATE = textwrap.dedent("""
        ### Instructions: 
        We are seeking your help to find an answer to this problem:
        The following is a prompt that was given to an AI assistant, and its corresponding response. 

        ### Here is the prompt:
        {prompt}

        ### Original Response:
        {original_response}
        
        We are considering making the following edits to the response:

        ### Revisions:
        {revisions}
        
        ### Final Response:
        {mutated_response}

        ### Instructions:
        We want to know if these revisions will lead to a loss in quality compared to the original.
        It is fine if some ideas are expressed differently, but we want to avoid introducing errors into the response.
        Be strict in your evaluation and consider the overall quality of the response, and take note of the differences between the two.
        If the revisions are acceptable, respond with "Yes", and if not, "No".

        Answer: 
    """)

    def format(self, row: dict) -> dict:

        # user_content
        revisions = find_revisions(row['original_response'], row['mutated_response'])
        row.update({"revisions": revisions})
        user_content = self.TEMPLATE.format(**row)

        # asst_content
        asst_content = yesno(row)

        # format output
        system_message = {"role": "system", "content": self.system_message}
        user_message   = {"role": "user", "content": user_content}
        asst_message   = {"role": "assistant", "content": asst_content}
        messages       = [system_message, user_message, asst_message]

        return {"messages": messages}

##################
# MutationOracle #
##################

class MutationOracleFormatter(BaseFormatter):
    TEMPLATE = textwrap.dedent("""
        ### Instructions: 
        We are seeking your help to find an answer to this problem:
        The following is a prompt that was given to an AI assistant, and its corresponding response. 
        After that, is the same answer, but rephrased. 

        ### Here is the prompt:
        {prompt}

        ### Original Response:
        {original_response}             

        ### Rephrased Response:
        {mutated_response}

        ### Instructions: 
        We want to know if the rephrased answer maintains the same level of quality and accuracy as the original.
        Please make your evaluation based on the level of grammatical correctness, fluency, accuracy, structure, and clarity in the original vs rephrased answers.
        Be strict in your evaluation and consider the overall quality of the response, and take note of the differences between the two.
        If the rephrased response is just as good or better, output "Yes", otherwise output "No".

        Answer:
    """)

    def format(self, row: dict) -> dict:

        # user_content
        user_content = self.TEMPLATE.format(**row)

        # asst_content
        asst_content = yesno(row)

        # format output
        system_message = {"role": "system", "content": self.system_message}
        user_message   = {"role": "user", "content": user_content}
        asst_message   = {"role": "assistant", "content": asst_content}
        messages       = [system_message, user_message, asst_message]

        return {"messages": messages}

if __name__ == "__main__":

    import pandas as pd

    df = pd.read_csv("./data/IMP/train.csv")

    formatter = DiffOracleFormatter()

    formatted_df = formatter.apply_to_dataframe(df)
    # formatted_df.to_csv("./data/IMP/train_openai_formatted.csv")

    print(formatted_df)
