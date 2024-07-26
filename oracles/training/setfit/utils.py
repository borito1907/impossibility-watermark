import textwrap
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, 
    roc_auc_score, hamming_loss, f1_score, jaccard_score, 
)

WQE_INPUT_TEMPLATE = textwrap.dedent("""### Here is the prompt: 
    {{instruction}} 

    ### Model A Response: 
    {{response_A}} 

    ### Model B Response: 
    {{response_B}} 

    ### Instructions: 
    Compare which of the two above responses is a better response to the given prompt. 
    Your answer should be chosen from the following three options:
        0: Response A is better than response B
        1: Response B is better than response A
        2: Responses A and B have similar quality
    Please avoid any potential bias and ensuring that the order in which the responses were presented does not affect your judgment. """)


def prepare_dataset(dataset):
    # prompt,response_a,response_b,winner_model_a,winner_model_b,winner_tie
    def format_row(row):
        text = WQE_INPUT_TEMPLATE\
            .replace("{{instruction}}", row["prompt"])\
            .replace("{{response_A}}", row["response_a"])\
            .replace("{{response_B}}", row["response_b"])
        label = 2 if row["winner_tie"] else 0 if row["winner_model_a"] else 1
        return {"text" : text, "label": label}

    dataset = dataset.map(format_row)
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in ["text", "label"]])
    return dataset

def compute_metrics(y_pred, y_test):
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'hamming_accuracy': 1-hamming_loss(y_test, y_pred),
        'recall': recall_score(y_test, y_pred, average='micro', zero_division=0),
        'precision': precision_score(y_test, y_pred, average='micro', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='micro', zero_division=0),
        'jaccard_score': jaccard_score(y_test, y_pred, average='micro', zero_division=0),
        'support': len(y_pred)
    }
    return metrics