import pandas as pd
from sklearn.metrics import confusion_matrix

def calculate_weighted_metrics(TP, FP, FN, TN):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
    return precision, recall, f1_score, accuracy

def analyze_response_quality(file_paths, penalty_weights=None):
    if penalty_weights is None:
        penalty_weights = {
            "A_BETTER": 1.0,
            "B_BETTER": 1.0,
            "TIE": 1.0
        }

    # Load the dataset
    keys = ['oracle_type', 'oracle_class', 'judge_name', 'explain']
    others = ['time_taken']
    vals = ['original_label', 'original_pred']
    cols = keys + others + vals

    # Load and concatenate the datasets
    df_list = [pd.read_csv(f, encoding='ISO-8859-1') for f in file_paths]
    df = pd.concat(df_list, axis=0, ignore_index=True)

    # Ensure the dataframe contains only the columns specified in 'cols'
    df = df[cols]

    # Mapping enum string values to integer labels
    mapping = {
        "ResponseQuality.A_BETTER": 0,
        "ResponseQuality.B_BETTER": 1,
        "ResponseQuality.TIE": 2
    }

    # Apply the mapping to the relevant columns
    df['original_label'] = df['original_label'].map(mapping)
    df['original_pred'] = df['original_pred'].map(mapping)

    # Fill NaN values with a placeholder to ensure they are included in the grouping
    df['judge_name'] = df['judge_name'].fillna(df['oracle_class'])

    # Handle missing values 
    df_clean = df.dropna(subset=vals)
    
    def calculate_metrics(x):
        # Reverse mapping for confusion matrix labels
        reverse_mapping = {v: k.replace("ResponseQuality.", "") for k, v in mapping.items()}
        
        # Calculate confusion matrix using string labels
        cm = confusion_matrix(
            x['original_label'], 
            x['original_pred'], 
            labels=list(mapping.values())
        )
        
        class_metrics = {}
        total_support = 0
        weighted_precision_sum = 0
        weighted_recall_sum = 0
        weighted_f1_sum = 0
        weighted_accuracy_sum = 0
        
        for i, class_label in reverse_mapping.items():
            # Apply weights to TP, FP, FN, TN
            weight = penalty_weights[class_label]
            TP = cm[i, i] * weight
            FP = (cm[:, i].sum() - cm[i, i]) * weight
            FN = (cm[i, :].sum() - cm[i, i]) * weight
            
            # True Negatives (TN) = Total samples - (TP + FP + FN)
            TN = (cm.sum() - (TP + FP + FN)) * weight
            
            # Calculate weighted precision, recall, f1-score, and accuracy
            precision, recall, f1_score, accuracy = calculate_weighted_metrics(TP, FP, FN, TN)
            
            # Support is the total number of true instances of the class
            support = cm[i, :].sum()
            
            # Accumulate the weighted sums for averaging
            weighted_precision_sum += precision * support
            weighted_recall_sum += recall * support
            weighted_f1_sum += f1_score * support
            weighted_accuracy_sum += accuracy * support
            total_support += support
            
            # Store TP, TN, FP, FN
            class_metrics[f'TP_{class_label}'] = TP
            class_metrics[f'TN_{class_label}'] = TN
            class_metrics[f'FP_{class_label}'] = FP
            class_metrics[f'FN_{class_label}'] = FN

            # Store the metrics for each class
            class_metrics[f'support_{class_label}'] = support
            class_metrics[f'precision_{class_label}'] = precision
            class_metrics[f'recall_{class_label}'] = recall
            class_metrics[f'f1_score_{class_label}'] = f1_score
            class_metrics[f'accuracy_{class_label}'] = accuracy
        
        # Calculate overall weighted average metrics
        avg_metrics = {
            'average_time_taken': x['time_taken'].mean(),
            'average_precision': weighted_precision_sum / total_support if total_support > 0 else 0,
            'average_recall': weighted_recall_sum / total_support if total_support > 0 else 0,
            'average_f1_score': weighted_f1_sum / total_support if total_support > 0 else 0,
            'average_accuracy': weighted_accuracy_sum / total_support if total_support > 0 else 0,
        }
        
        return pd.Series({**avg_metrics, **class_metrics})
    
    # Group the data by specified columns and calculate metrics for each group
    grouped_metrics = df_clean.groupby(keys).apply(calculate_metrics).reset_index()
    
    return grouped_metrics.sort_values("average_f1_score")

file_paths = [
    "./oracles/results/IMP_oracle_eval_gpt-4o-ft.csv",
    './oracles/results/IMP_oracle_eval_v3.csv',
    './oracles/results/IMP_oracle_eval_DiffOracle-IMP-sft.csv',
    './oracles/results/IMP_oracle_eval_sfts.csv',
    './oracles/results/IMP_oracle_eval_nicol2.csv'
]

# Example usage with different penalty weights for each class
penalty_weights = {
    "A_BETTER": 1.0,
    "B_BETTER": 1.0,
    "TIE": 1.0
}

results = analyze_response_quality(file_paths, penalty_weights=penalty_weights)
print(results)
results.to_csv('./oracles/results/IMP_oracle_eval_all_metrics2.csv', index=False)
