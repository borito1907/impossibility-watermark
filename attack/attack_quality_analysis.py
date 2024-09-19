import pandas as pd
from extractors import FluencyMetric, GrammarMetric, QualityMetric
import os
import matplotlib.pyplot as plt


def generate_step_data(file):
  attack_trace_df = pd.read_csv(file)

  fluency_metric = FluencyMetric()
  grammar_metric = GrammarMetric()
  quality_metric = QualityMetric()


  # include initial texts
  initial_attack_df = attack_trace_df[attack_trace_df['step_num'] == 0]
  step_data = {'count': 0, "avg_quality": 0, "avg_fluency": 0, "avg_grammar": 0}
  
  step_data['count'] = len(initial_attack_df)
  step_data["avg_quality"] = quality_metric.evaluate(initial_attack_df['prompt'], initial_attack_df['current_text'])
  step_data["avg_fluency"] = fluency_metric.evaluate(initial_attack_df['current_text'])
  step_data["avg_grammar"] = grammar_metric.evaluate(initial_attack_df['current_text'])
  
  # lsit to hold data
  attack_data_per_step = [step_data]


	# only care about times where mutations were successful
  successful_mutation_df = attack_trace_df[(attack_trace_df["quality_preserved"] == True) & (attack_trace_df["length_issue"] == False)]
	# separate by mutation_num
  successful_mutation_df = successful_mutation_df.groupby('mutation_num')

  # evaluate in groups of mutation_num (this does not include initial texts)
  for name, group in successful_mutation_df:
    step_data = {'count': 0, "avg_quality": 0, "avg_fluency": 0, "avg_grammar": 0} 

    step_data['count'] = len(group)
    step_data["avg_quality"] = quality_metric.evaluate(group['prompt'], group['mutated_text'])
    step_data["avg_fluency"] = fluency_metric.evaluate(list(group['mutated_text']))
    step_data["avg_grammar"] = grammar_metric.evaluate(group['mutated_text'])

    attack_data_per_step.append(step_data)


  return pd.DataFrame(attack_data_per_step)




def graph_data(data_df, image_file):
	# Graphs:

	# Quality score vs nth successful step
	# Fluency vs nth successful step
	# Grammar errors vs nth successful step
  # Number of instancs vs nth successful step

  fig, ax = plt.subplots(2, 2, figsize=(12, 15))

  # Quality
  ax[0][0].plot(data_df.index, data_df['avg_quality'], label="quality")
  ax[0][0].set_title(f"Average Quality Score VS Successful Steps")
  ax[0][0].set_xlabel("Successful Mutations")
  ax[0][0].set_ylabel("Average InternLM Quality Score")

  # Perplexity
  ax[0][1].plot(data_df.index, data_df['avg_fluency'], label="fluency")
  ax[0][1].set_title(f"Average Fluency VS Successful Steps")
  ax[0][1].set_xlabel("Successful Mutations")
  ax[0][1].set_ylabel("Average Fluency Score")

  # Grammar
  ax[1][0].plot(data_df.index, data_df['avg_grammar'], label="grammar")
  ax[1][0].set_title(f"Average Grammar Errors VS Successful Steps")
  ax[1][0].set_xlabel("Successful Mutations")
  ax[1][0].set_ylabel("Average Grammar Errors Count")

  # Counts
  ax[1][1].plot(data_df.index, data_df['count'], label="count")
  ax[1][1].set_title(f"Number of Samples VS Successful Steps")
  ax[1][1].set_xlabel("Successful Mutations")
  ax[1][1].set_ylabel("Number of Samples")

  #plt.show()
  plt.savefig(image_file)





if __name__ == "__main__":
  file = "attack_traces/DiffOracle_UMDWatermarker_DocumentMutator_1step_compare-original=False_attack_results.csv"

  data_file = file.split("/")[0] + "/" + "quality_" + file.split("/")[1]
  # generate data if it doesnt exist already
  if not os.path.isfile(data_file):
    data_df = generate_step_data(file)
    data_df.to_csv(data_file)
  # capture data
  else:
    data_df = pd.read_csv(data_file)
  
  image_file = data_file.split(".csv")[0] + ".png"
  graph_data(data_df, image_file)