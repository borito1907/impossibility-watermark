import ast
from functools import partial
from datasets import concatenate_datasets

def validate_utf8_compliance(dataset):
    # Identify all string columns in the dataset
    string_columns = [column for column, dtype in dataset.features.items() if dtype == 'string']
    
    # Function to validate and possibly fix UTF-8 encoding for a single example
    def validate_example(example):
        for column in string_columns:
            try:
                # Attempt to encode and decode to ensure UTF-8 compliance
                example[column] = example[column].encode('utf-8').decode('utf-8')
            except UnicodeEncodeError:
                # Handle strings that are not UTF-8 compliant
                example[column] = example[column].encode('ascii', errors='ignore').decode('utf-8')
        return example
    
    # Apply the validation function to all examples in the dataset
    return dataset.map(validate_example)

# Filter dataset by word length and response turns
def filter(dataset):
    # Filter by word length
    output_low, output_high = 100, 1000
    prompt_low, prompt_high = 0, 20
    dataset = dataset.filter(lambda example: prompt_low <= len(example["prompt"].split()) <= prompt_high)
    dataset = dataset.filter(lambda example: output_low <= len(example["response_a"].split()) <= output_high)
    dataset = dataset.filter(lambda example: output_low <= len(example["response_b"].split()) <= output_high)

    # Keep only single turn conversations
    num_turns = 1
    dataset = dataset.filter(lambda example: len(ast.literal_eval(example["prompt"])) == num_turns)

    return dataset

# Select number of samples for each winner type
def select(dataset, winner_a_amount, winner_b_amount, winner_tie_amount):
    winner_model_a = dataset.filter(lambda x: x['winner_model_a'] == 1).select(range(winner_a_amount))
    winner_model_b = dataset.filter(lambda x: x['winner_model_b'] == 1).select(range(winner_b_amount))
    winner_tie     = dataset.filter(lambda x: x['winner_tie']     == 1).select(range(winner_tie_amount))
    dataset = concatenate_datasets([winner_model_a,winner_model_b,winner_tie]).shuffle()

    return dataset

# Helper function to convert list representations into string
def convert_list_to_str(example, key="prompt"):
    try:
        # Evaluate the string representation of the list
        content = ast.literal_eval(example[key])
        if isinstance(content, list) and content:
            # Handle surrogate pairs properly, replace errors with a placeholder or remove them
            result = content[0].encode('utf-8', errors='replace').decode('utf-8')
            return {key: result}
        else:
            # Return the original string or a placeholder if the list is empty
            return {key: "Empty or invalid list"} if not content else {key: content[0]}
    except (SyntaxError, ValueError) as e:
        # Return None or log error, and provide an informative message
        return {key: f"Error parsing list: {str(e)}"}
    except UnicodeEncodeError as e:
        # Handle Unicode encoding errors explicitly
        return {key: f"Unicode error: {str(e)}"}
    
# Reformat dataset columns to strings
def reformat(dataset):
    dataset = dataset.map(partial(convert_list_to_str, key="prompt"))
    dataset = dataset.map(partial(convert_list_to_str, key="response_a"))
    dataset = dataset.map(partial(convert_list_to_str, key="response_b"))
    return dataset

def preprocess(dataset):
    dataset = validate_utf8_compliance(dataset)
    # Filters for one-turn conversations and response/prompt length
    # Modify the lengths directly in the function
    dataset = filter(dataset)
    print(f"Dataset after length filtering: {dataset}")
    # Selects samples from winner_a, winner_b, and winner_tie
    # dataset = select(dataset, 5000, 5000, 5000)
    # print(f"Dataset after winner selection: {dataset}")
    dataset = reformat(dataset)
    return dataset
