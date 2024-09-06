from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import math
tests_df = pd.read_csv("./results/mutator_eval.csv")
data = {}
output = []

for index, row in tqdm(tests_df.iterrows(), desc='Tests'):
    try:
        float(row["fluency_score"])
    except:
        continue
    if row["mutator"] not in data:
        data[row["mutator"]] = {row["mutation_step"]: {"total": 0, "quality": 0, "fluency": 0, "grammar": 0}}
    if row["mutation_step"] not in data[row["mutator"]]:
        data[row["mutator"]][row["mutation_step"]] = {"total": 0, "quality": 0, "fluency": 0, "grammar": 0}
    data[row["mutator"]][row["mutation_step"]]["total"] += 1

    data[row["mutator"]][row["mutation_step"]]["fluency"] += float(row["fluency_score"])
    if math.isnan(float(row["count_grammar_errors"])):
        data[row["mutator"]][row["mutation_step"]]["grammar"] += 0
    else: 
        data[row["mutator"]][row["mutation_step"]]["grammar"] += float(row["count_grammar_errors"])
    if row["quality_preserved"] == "True":
        data[row["mutator"]][row["mutation_step"]]["quality"] += 1
print(data)
# for i in data:
#     for j in data[i]:
#         temp = {}
#         temp["mutator"] = i
#         temp["mutation step"] = j
#         temp["percent preserved"] = data[i][j][0]/ data[i][j][1]
#         output.append(temp)
df = pd.DataFrame(output)
df.to_csv("./results/mutator_percent.csv")
fig, ax = plt.subplots(3, len(data.keys()), figsize=(20, 25))
for i, mut in enumerate(data.keys()):
    d = {i: data[mut][i]['quality']/data[mut][i]['total'] for i in data[mut]}
    ax[0][i].plot(d.keys(), d.values(), label="quality preserved")
    ax[0][i].set_title(f"Percentage vs\nsteps for {mut}")
    ax[0][i].set_xlabel("Number of mutation steps")
    ax[0][i].set_ylabel("Percentage of \n quality preserved")
    ax[0][i].set_ylim([0,1.2])
    d = {i: data[mut][i]['fluency']/data[mut][i]['total'] for i in data[mut]}
    ax[1][i].plot(d.keys(), d.values(), label="quality preserved")
    ax[1][i].set_title(f"Percentage vs\nsteps for {mut}")
    ax[1][i].set_xlabel("Number of mutation steps")
    ax[1][i].set_ylabel("Average \n fluency score")
    ax[1][i].set_ylim([0,300])
    d = {i: data[mut][i]['grammar']/data[mut][i]['total'] for i in data[mut]}
    ax[2][i].plot(d.keys(), d.values(), label="quality preserved")
    ax[2][i].set_title(f"Percentage vs\nsteps for {mut}")
    ax[2][i].set_xlabel("Number of mutation steps")
    ax[2][i].set_ylabel("Average \n Grammar errors")
    ax[2][i].set_ylim([0,36])
# fig.tight_layout(pad=5.0)
plt.show()
plt.savefig("mutator.png")