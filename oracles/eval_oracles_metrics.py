import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report

def analyze_response_quality(file_paths):
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

    for key in keys:
        print(key, df[key].unique())

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
        # Calculate average metrics
        avg_metrics = {
            'average_time_taken': x['time_taken'].mean(),
            'overall_accuracy': accuracy_score(x['original_label'], x['original_pred']),
            'overall_recall': recall_score(x['original_label'], x['original_pred'], average='weighted'),
            'overall_precision': precision_score(x['original_label'], x['original_pred'], average='weighted'),
            'overall_f1_score': f1_score(x['original_label'], x['original_pred'], average='weighted'),
        }
        
        # Calculate metrics for each class
        report = classification_report(
            y_true=x['original_label'], 
            y_pred=x['original_pred'], 
            labels=list(mapping.values()),
            target_names=list(mapping.keys()),
            output_dict=True, 
            zero_division=0)
        
        class_metrics = {}
        for class_label in list(mapping.keys()):  # Specific class labels expected in the report
            class_metrics[f'class_{class_label.replace("ResponseQuality.", "")}_precision'] = report[class_label]['precision']
            class_metrics[f'class_{class_label.replace("ResponseQuality.", "")}_recall'] = report[class_label]['recall']
            class_metrics[f'class_{class_label.replace("ResponseQuality.", "")}_f1_score'] = report[class_label]['f1-score']
            class_metrics[f'class_{class_label.replace("ResponseQuality.", "")}_support'] = report[class_label]['support']
        
        # Combine average metrics and class-specific metrics
        return pd.Series({**avg_metrics, **class_metrics})
    
    # Group the data by specified columns and calculate metrics for each group
    grouped_metrics = df_clean.groupby(keys).apply(calculate_metrics).reset_index()
    
    return grouped_metrics.sort_values("overall_f1_score")

file_paths = [
    './oracles/results/IMP_oracle_eval_v2.csv',
    './oracles/results/IMP_oracle_eval_DiffOracle-IMP-sft.csv'
]
results = analyze_response_quality(file_paths)
print(results)
results.to_csv('./oracles/results/IMP_oracle_eval_v2_metrics.csv', index=False)
