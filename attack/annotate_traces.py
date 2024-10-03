import pandas as pd

def assign_unique_group_ids(df):
    df['new_group'] = (df['step_num'] == 0).astype(int)
    df['group_id'] = df['new_group'].cumsum()
    return df

def get_support(df):
    return (df['step_num'] == 0).sum()

def get_max_step_count(df):
    return df['step_num'].max()

if __name__ == "__main__":

    # RUN: CUDA_VISIBLE_DEVICES=0 python -m attack.annotate_traces

    import os
    import glob
    from extractors import FluencyMetric, GrammarMetric, QualityMetric, EditsMetric
        
    # Initialize metric extractors
    fluency = FluencyMetric()
    grammar = GrammarMetric()
    quality = QualityMetric()
    edits   = EditsMetric()

    traces = glob.glob("./attack_traces/*attack_results.csv")

    for trace in traces:

        o, w, m, s = os.path.basename(trace).split("_")[:4]
        s = int(s.replace("n-steps=", ""))
        
        df = assign_unique_group_ids(pd.read_csv(trace)).head(5)
        df["mutated_text"] = df["mutated_text"].fillna(df["current_text"])
        df['current_text'] = df['mutated_text'].shift(1)
        df["current_text"] = df["current_text"].fillna(df["mutated_text"])

        # step_num,mutation_num,prompt,current_text,mutated_text,current_text_len,mutated_text_len,length_issue,quality_analysis,quality_preserved,watermark_detected,watermark_score,backtrack,total_time,mutator_time,oracle_time
        if "edit_count" not in df.columns:
            df = edits.evaluate_dataframe(df, texts1_column="current_text", texts2_column="mutated_text", new_column="perplexity")
        if "perplexity_score" not in df.columns:
            df = fluency.evaluate_dataframe(df, text_column="mutated_text", new_column="perplexity")
        if "grammar_score" not in df.columns:    
            df = grammar.evaluate_dataframe(df, text_column="mutated_text", new_column="grammar_errors")
        if "internlm_quality_score" not in df.columns:
            df = quality.evaluate_dataframe(df, prompt_column="prompt", text_column="mutated_text", new_column="internlm_quality")

        df.to_csv(trace.replace("attack_results.csv", "attack_results_annotated.csv"), index=False)
        print(df)