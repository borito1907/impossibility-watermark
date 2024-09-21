import pandas as pd
from extractors import FluencyMetric, GrammarMetric, QualityMetric
import os
import random
import matplotlib.pyplot as plt


# quality_metric = QualityMetric()
# perplexity_metric = FluencyMetric()
# grammar_metric = GrammarMetric()


def generate_data(file):
  attack_trace_df = pd.read_csv(file)

  # only care about times where mutations were successful
  successful_mutation_df = attack_trace_df[(attack_trace_df["quality_preserved"] == True) & (attack_trace_df["length_issue"] == False)]
  #successful_mutation_df = successful_mutation_df[['mutation_num', 'prompt', 'current_text', 'mutated_text', 'watermark_score']]

  # separate by prompt, add initial text as step -1
  # TODO: need initial watermark score. until then, just dont include initial step
  # for name, group in successful_mutation_df.groupby('prompt'):
  #   group.iloc[-1] = [-1, group.iloc[0]['prompt'], group.iloc[0]['current_text'], group.iloc[0]['current_text'], something]
  #   group.index = group.index + 1
  #   group.sort_index()
    
  # contains mimimum necessary data
  successful_mutation_df = successful_mutation_df[['mutation_num', 'prompt', 'mutated_text', 'watermark_score']]

  # compute metrics
  # successful_mutation_df['quality'] = successful_mutation_df.apply(lambda row: quality_metric.evaluate([row['prompt']], [row['mutated_text']]), axis=1)
  # successful_mutation_df['perplexity'] = successful_mutation_df.apply(lambda row: perplexity_metric.evaluate([row['mutated_text']]), axis=1)
  # successful_mutation_df['grammar'] = successful_mutation_df.apply(lambda row: perplexity_metric.evaluate([row['mutated_text']]), axis=1)

  # batch compute
  successful_mutation_df['quality'] = quality_metric.evaluate(successful_mutation_df['prompt'], successful_mutation_df['mutated_text'], return_mean=False)
  successful_mutation_df['perplexity'] = perplexity_metric.evaluate(list(successful_mutation_df['mutated_text']), return_mean=False)
  successful_mutation_df['grammar'] = grammar_metric.evaluate(successful_mutation_df['mutated_text'], return_mean=False)

  return successful_mutation_df




def graph_data(data_df, image_file):
  # Graphs:

  # Quality score vs nth successful step
  # Fluency vs nth successful step
  # Grammar errors vs nth successful step

  fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 10))

  max_steps = data_df['mutation_num'].max()+1

  # separate by prompt
  for name, group in data_df.groupby('prompt'):
    #alpha = len(group)/max_steps
    lowest_watermark = group['watermark_score'].min()
    group['quality'] -= group.iloc[0]['quality']
    group['perplexity'] -= group.iloc[0]['perplexity']
    group['grammar'] -= group.iloc[0]['grammar']
  
    if lowest_watermark < 0:
      ax1.plot(group['mutation_num'], group['quality'], linewidth=1.2, label=name[:8])
      ax2.plot(group['mutation_num'], group['perplexity'], linewidth=1.2, label=name[:8])
      ax3.plot(group['mutation_num'], group['grammar'], linewidth=1.2, label=name[:8])
    else:
      ax1.plot(group['mutation_num'], group['quality'], linewidth=.6, label=name[:8], color="silver")
      ax2.plot(group['mutation_num'], group['perplexity'], linewidth=.6, label=name[:8], color="silver")
      ax3.plot(group['mutation_num'], group['grammar'], linewidth=.6, label=name[:8], color="silver")


  # Quality
  ax1.set_title(f"Change in Quality Score VS Successful Steps")
  ax1.set_xlabel("Successful Mutations")
  ax1.set_ylabel("Change in InternLM Quality Scoree")

  # Perplexity
  ax2.set_title(f"Change in Perplexity VS Successful Steps")
  ax2.set_xlabel("Successful Mutations")
  ax2.set_ylabel("Change in Perplexity Score")

  # Grammar
  ax3.set_title(f"Change in Grammar Errors VS Successful Steps")
  ax3.set_xlabel("Successful Mutations")
  ax3.set_ylabel("Change in Grammar Errors Count")
  #plt.show()
  plt.savefig(image_file)





if __name__ == "__main__":
  files = ["attack_traces/DiffOracle_UMDWatermarker_DocumentMutator_compare-original=False_200_attack_results.csv",
           #"attack_traces/DiffOracle_UMDWatermarker_DocumentMutator_1step_compare-original=False_200_attack_results.csv",
           "attack_traces/DiffOracle_UMDWatermarker_SentenceMutator_compare-original=False_200_attack_results.csv",
           "attack_traces/DiffOracle_UMDWatermarker_SentenceMutator_compare-original=True_200_attack_results.csv",
           "attack_traces/DiffOracle_UMDWatermarker_SpanMutator_compare-original=False_200_attack_results.csv",
           "attack_traces/DiffOracle_UMDWatermarker_SpanMutator_compare-original=True_200_attack_results.csv",
           "attack_traces/DiffOracle_UMDWatermarker_WordMutator_compare-original=False_200_attack_results.csv",
           "attack_traces/DiffOracle_UMDWatermarker_WordMutator_compare-original=True_200_attack_results.csv",]
  
  for file in files:

    data_file = file.split("/")[0] + "/quality/" + "quality_" + file.split("/")[1]
    # generate data if it doesnt exist already
    if not os.path.isfile(data_file):
      data_df = generate_data(file)
      data_df.to_csv(data_file, index=False)
    
    data_df = pd.read_csv(data_file)
    
    image_file = data_file.split(".csv")[0] + ".png"
    graph_data(data_df, image_file)