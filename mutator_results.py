from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
tests_df = pd.read_csv("./results/mutator_watermark_eval.csv")
data = {}
output = []
for index, row in tqdm(tests_df.iterrows(), desc='Tests'):
    if row["mutator"] not in data:
        data[row["mutator"]] = {row["mutation_step"]: [0,0]}
    if row["mutation_step"] not in data[row["mutator"]]:
        data[row["mutator"]][row["mutation_step"]] = [0,0]
    data[row["mutator"]][row["mutation_step"]][1] += 1
    data[row["mutator"]][row["mutation_step"]][0] += row['score']
print(data)
for i in data:
    for j in data[i]:
        temp = {}
        temp["mutator"] = i
        temp["mutation step"] = j
        temp["percent preserved"] = data[i][j][0]/ data[i][j][1]
        output.append(temp)
df = pd.DataFrame(output)
df.to_csv("./results/mutator_watermark_percent.csv")

fig, ax = plt.subplots(len(data))
for i, mut in enumerate(data.keys()):
    ax[i].plot(data[mut].keys(), [i[0]/i[1] for i in data[mut].values()])
    ax[i].set_title(f"Watermark Z-Score vs steps for {mut}")
    ax[i].set_xlabel("Number of mutation steps")
    ax[i].set_ylabel("Average score")
    ax[i].set_ylim([0, 7.2])
    ax[i].set_yticks([0,2,4,6])
fig.tight_layout(pad=5.0)
plt.show()
plt.savefig("mutator.png")