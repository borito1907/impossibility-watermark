import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import os

if __name__ == "__main__":
    traces1 = glob.glob("./attack_traces/*total.csv")
    data = {}
    for trace in traces1:
        w = os.path.basename(trace).split("_")[0]
        data[w] = {}
        df = pd.read_csv(trace)
        t = 0
        start = 0
        prev = 0
        prev_step = 0
        for idx, row in df.iterrows():
            if idx == 0 or t != row['id']:
                t = row['id']
                start = row['watermark_score']
            if row['step_num'] in [50, 200, 1000]:
                if row['mutator'] in data[w] and row['domain'] in data[w][row['mutator']]:
                    data[w][row['mutator']][row['domain']][0] += row['watermark_score'] - start
                    data[w][row['mutator']][row['domain']][1] += 1
                elif row['mutator'] in data[w]:
                    data[w][row['mutator']][row['domain']] = [row['watermark_score'] - start, 1]
                else:
                    data[w][row['mutator']] = {}
                    data[w][row['mutator']][row['domain']] = [row['watermark_score'] - start, 1]
    print(data)
    for watermark in data.keys():
        mutators = data[watermark].keys()
        fig, axs = plt.subplots(len(mutators), figsize = (15, 25))
        for i, mutator in enumerate(mutators):
            keys = [i for i in data[watermark][mutator].keys() if not i != i]
            print(keys)
            axs[i].barh(keys, [int(data[watermark][mutator][i][0]/ data[watermark][mutator][i][1]) for i in keys])
            axs[i].set_title(f"{mutator}")
            axs[i].set_ylabel("Z-score drop")
            axs[i].set_xlabel("Domains")

        plt.savefig(f"./attack/graphs/{watermark}_domain_analysis.png")
        plt.clf()
    
    data = {}
    for trace in traces1:
        w = os.path.basename(trace).split("_")[0]
        data[w] = {}
        df = pd.read_csv(trace)
        t = 0
        start = 0
        prev = 0
        prev_step = 0
        for idx, row in df.iterrows():
            if idx == 0 or t != row['id']:
                t = row['id']
                start = row['watermark_score']
            if row['step_num'] in [50, 200, 1000]:
                if row['mutator'] in data[w] and row['entropy'] in data[w][row['mutator']]:
                    data[w][row['mutator']][row['entropy']][0] += row['watermark_score'] - start
                    data[w][row['mutator']][row['entropy']][1] += 1
                elif row['mutator'] in data[w]:
                    data[w][row['mutator']][row['entropy']] = [row['watermark_score'] - start, 1]
                else:
                    data[w][row['mutator']] = {}
                    data[w][row['mutator']][row['entropy']] = [row['watermark_score'] - start, 1]
    print(data)
    for watermark in data.keys():
        mutators = data[watermark].keys()
        fig, axs = plt.subplots(len(mutators), figsize = (15, 25))
        for i, mutator in enumerate(mutators):
            keys = [i for i in data[watermark][mutator].keys() if not i != i]
            print(keys)
            axs[i].barh(keys, [int(data[watermark][mutator][i][0]/ data[watermark][mutator][i][1]) for i in keys])
            axs[i].set_title(f"{mutator}")
            axs[i].set_ylabel("Z-score drop")
            axs[i].set_xlabel("Entropy")

        plt.savefig(f"./attack/graphs/{watermark}_entropy_analysis.png")
        plt.clf()