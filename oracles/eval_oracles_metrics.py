import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def analyze_response_quality(file_path):
    # Load the dataset
    keys = ['oracle_type', 'oracle_class', 'judge_name', 'explain']
    others = ['time_taken']
    vals = ['original_label', 'original_pred']
    cols = keys + others + vals

    df = pd.read_csv(file_path, encoding='ISO-8859-1')[cols]

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
    
    # Group the data by specified columns and calculate metrics for each group
    grouped_metrics = df_clean.groupby(keys).apply(
        lambda x: pd.Series({            
            'average_time_taken': x['time_taken'].mean(),
            'accuracy': accuracy_score(x['original_label'], x['original_pred']),
            'recall': recall_score(x['original_label'], x['original_pred'], average='weighted'),
            'precision': precision_score(x['original_label'], x['original_pred'], average='weighted'),
            'f1_score': f1_score(x['original_label'], x['original_pred'], average='weighted'),
        })
    ).reset_index()
    
    return grouped_metrics.sort_values("f1_score")

file_path = './oracles/results/IMP_oracle_eval_v2.csv'
results = analyze_response_quality(file_path)
print(results)
results.to_csv('./oracles/results/IMP_oracle_eval_v2_metrics.csv', index=False)
