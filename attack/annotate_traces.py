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

    # RUN: CUDA_VISIBLE_DEVICES=3,4 python -m attack.annotate_traces

    import os
    import glob
    import traceback
    from extractors import FluencyMetric, GrammarMetric, InternLMQualityMetric, EditsMetric
        
    # Initialize metric extractors
    fluency = FluencyMetric()
    grammar = GrammarMetric()
    quality = InternLMQualityMetric()
    edits   = EditsMetric()

    traces = glob.glob("./attack_traces/*attack_results_annotated.csv")

    for trace in traces:

        print(trace)

        o, w, m, s = os.path.basename(trace).split("_")[:4]
        print(s)
        s = int(s.replace("n-steps=", ""))
        
        df = assign_unique_group_ids(pd.read_csv(trace))
        df["mutated_text"] = df["mutated_text"].fillna(df["current_text"])
        df['current_text'] = df['mutated_text'].shift(1)
        df["current_text"] = df["current_text"].fillna(df["mutated_text"])

        # step_num,mutation_num,prompt,current_text,mutated_text,current_text_len,mutated_text_len,length_issue,quality_analysis,quality_preserved,watermark_detected,watermark_score,backtrack,total_time,mutator_time,oracle_time
        if "words_edited" not in df.columns:
            try:
                df = edits.evaluate_dataframe(df, current_text_column="current_text", mutated_text_column="mutated_text", new_column="words_edited")
            except:
                print(f"{'=' * 50} words_edited {'=' * 50}")
                print(traceback.format_exc())
        if "perplexity" not in df.columns:
            try:
                df = fluency.evaluate_dataframe(df, text_column="mutated_text", new_column="perplexity")
            except:
                print(f"{'=' * 50} perplexity {'=' * 50}")
                print(traceback.format_exc())
        if "grammar_errors" not in df.columns:    
            try:
                df = grammar.evaluate_dataframe(df, text_column="mutated_text", new_column="grammar_errors")
            except:
                print(f"{'=' * 50} grammar_errors {'=' * 50}")
                print(traceback.format_exc())
        if "internlm_quality" not in df.columns:
            try:
                df = quality.evaluate_dataframe(df, prompt_column="prompt", text_column="mutated_text", new_column="internlm_quality")
            except:
                print(f"{'=' * 50} internlm_quality {'=' * 50}")
                print(traceback.format_exc())

        print(df)
        print(trace)
        df.to_csv(trace, index=False)