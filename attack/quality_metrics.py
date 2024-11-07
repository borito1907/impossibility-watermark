import pandas as pd
import os
import glob
import random
import matplotlib.pyplot as plt


def graph_data(annotated_file):
  print(f"QUALITY_METRICS: Graphing {annotated_file}")

  data_df = pd.read_csv(annotated_file)
  data_df = data_df[(data_df["quality_preserved"] == True) & (data_df["length_issue"] == False) & (data_df["mutation_num"] != -1)]

  # Graphs:

  # Quality score vs nth successful step
  # Fluency vs nth successful step
  # Grammar errors vs nth successful step
  # Words edited vs nth successful step

  fig, (ax1, ax3, ax4, ax5) = plt.subplots(1, 4, figsize=(36, 10))

  # separate by prompt
  for name, group in data_df.groupby('prompt'):

    group['internlm_quality'] -= group.iloc[0]['internlm_quality']
    group['perplexity'] -= group.iloc[0]['perplexity']
    group['grammar_errors'] -= group.iloc[0]['grammar_errors']
    group['words_edited'] /= group.iloc[0]['current_text_len']
    #group['skywork_quality'] -= group.iloc[0]['skywork_quality']
  
    # color_trace = False
    # if 'Adaptive' in annotated_file:
    #   color_trace = group[group['watermark_score'] != -1.]['watermark_score'].min() < 60
    # elif 'SemStamp' in annotated_file:
    #   color_trace = group[group['watermark_score'] != -1.]['watermark_score'].min() < 0
    try:
      color_trace = group[group['normalized_watermark_score'] != -1.]['normalized_watermark_score'].min() < 0
    except:
      print(f"QUALITY_METRICS: No column 'normalized_watermark_score'")
      return

    if color_trace:
      ax1.plot(group['mutation_num'], group['internlm_quality'], linewidth=1.2, label=name[:8])
      #ax2.plot(group['mutation_num'], group['skywork_quality'], linewidth=1.2, label=name[:8])
      ax3.plot(group['mutation_num'], group['perplexity'], linewidth=1.2, label=name[:8])
      ax4.plot(group['mutation_num'], group['grammar_errors'], linewidth=1.2, label=name[:8])
      ax5.plot(group['mutation_num'], group['words_edited'], linewidth=1.2, label=name[:8])
    else:
      ax1.plot(group['mutation_num'], group['internlm_quality'], zorder=-1, linewidth=.6, label=name[:8], color="silver")
      #ax2.plot(group['mutation_num'], group['skywork_quality'], zorder=-1, linewidth=.6, label=name[:8], color="silver")
      ax3.plot(group['mutation_num'], group['perplexity'], zorder=-1, linewidth=.6, label=name[:8], color="silver")
      ax4.plot(group['mutation_num'], group['grammar_errors'], zorder=-1, linewidth=.6, label=name[:8], color="silver")
      ax5.plot(group['mutation_num'], group['words_edited'], zorder=-1, linewidth=.6, label=name[:8], color="silver")

  size=16

  # InternLM Quality
  ax1.set_title("Change in Quality Score VS Successful Steps (InternLM)", fontsize=size)
  ax1.set_xlabel("Successful Mutations", fontsize=size)
  ax1.set_ylabel("Change in InternLM Quality Scoree", fontsize=size)
  
  # Skywork Quality
  # ax2.set_title("Change in Quality Score VS Successful Steps (Skywork)", fontsize=size)
  # ax2.set_xlabel("Successful Mutations", fontsize=size)
  # ax2.set_ylabel("Change in Skywork Quality Scoree", fontsize=size)

  # Perplexity
  ax3.set_title("Change in Perplexity VS Successful Steps", fontsize=size)
  ax3.set_xlabel("Successful Mutations", fontsize=size)
  ax3.set_ylabel("Change in Perplexity Score", fontsize=size)

  # Grammar
  ax4.set_title("Change in Grammar Errors VS Successful Steps", fontsize=size)
  ax5.set_xlabel("Successful Mutations", fontsize=size)
  ax5.set_ylabel("Change in Grammar Errors Count", fontsize=size)

  # Words edited
  ax5.set_title("Percentage of Words Changed VS Successful Steps", fontsize=size)
  ax5.set_xlabel("Successful Mutations", fontsize=size)
  ax5.set_ylabel("Percentage of Words Changed", fontsize=size)

  
  # Title + Info
  analysis_info = annotated_file.split("attack_traces/")[1].split("_")[:4]
  plt.suptitle(f"Quality Analysis for {analysis_info[1]} using {analysis_info[0]} and {analysis_info[2]} for {analysis_info[3]}", fontsize=26)

  file_path, file_name = os.path.split(annotated_file)
  image_file = os.path.join(file_path, "quality", "quality_" + file_name[:-3] + "png")
  print(f"QUALITY_METRICS: Saving to file {image_file}")
  plt.savefig(image_file)


if __name__ == "__main__":
  traces = glob.glob("./attack_traces/*attack_results_annotated*.csv")

  for file in traces:
    graph_data(file)
    