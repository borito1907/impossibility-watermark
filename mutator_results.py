from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
tests_df = pd.read_csv("./results/mutator_eval.csv")
data = {}
output = []
for index, row in tqdm(tests_df.iterrows(), desc='Tests'):
    if row["mutator"] not in data:
        data[row["mutator"]] = {row["mutation_step"]: [0,0,0,0]}
    if row["mutation_step"] not in data[row["mutator"]]:
        data[row["mutator"]][row["mutation_step"]] = [0,0,0,0]
    data[row["mutator"]][row["mutation_step"]][0] += 1
    if row["quality_preserved"]:
        data[row["mutator"]][row["mutation_step"]][1] += 1
    data[row["mutator"]][row["mutation_step"]][2] += row["fluency_score"]
    data[row["mutator"]][row["mutation_step"]][3] += row["count_grammar_errors"]
print(data)
for i in data:
    for j in data[i]:
        temp = {}
        temp["mutator"] = i
        temp["mutation_step"] = j
        temp["percent_preserved"] = data[i][j][1]/ data[i][j][0]
        temp["fluency_score"] = data[i][j][2]/ data[i][j][0]
        temp["count_grammar_errors"] = data[i][j][3]/ data[i][j][0]
        output.append(temp)
df = pd.DataFrame(output)
df.to_csv("./results/mutator_watermark_percent.csv")
mut = list(data.keys())[0]
fig, ax = plt.subplots(3, figsize=(12, 15))

ax[0].plot(data[mut].keys(), [i[1]/i[0] for i in data[mut].values()], label="quality preserved")
ax[0].set_title(f"Percentage vs steps for {mut}")
ax[0].set_xlabel("Number of mutation steps")
ax[0].set_ylabel("Percentage of \n quality preserved")
# ax[0].set_ylim([0,1.2])
ax[1].plot(data[mut].keys(), [i[2]/i[0] for i in data[mut].values()], label="quality preserved")
ax[1].set_title(f"Percentage vs steps for {mut}")
ax[1].set_xlabel("Number of mutation steps")
ax[1].set_ylabel("Average \n fluency score")
# ax[1].set_ylim([0,1.2])
ax[2].plot(data[mut].keys(), [i[3]/i[0] for i in data[mut].values()], label="quality preserved")
ax[2].set_title(f"Percentage vs steps for {mut}")
ax[2].set_xlabel("Number of mutation steps")
ax[2].set_ylabel("Average \n Grammar errors")
ax[2].set_ylim([0,1.2])


# for i, mut in enumerate(data.keys()):
#     plt.plot(data[mut].keys(), [i[1]/i[0] for i in data[mut].values()], label="quality preserved")
#     plt.plot(data[mut].keys(), [i[2]/i[0] for i in data[mut].values()], label="fluency score")
#     plt.plot(data[mut].keys(), [i[3]/i[0] for i in data[mut].values()], label="grammar errors")
#     plt.legend(loc='best')
#     plt.title(f"Percentage vs steps for {mut}")
#     plt.xlabel("Number of mutation steps")
#     plt.ylabel("Percentages")
#     plt.ylim([0,1.2])
# fig.tight_layout(pad=5.0)

plt.show()
plt.savefig("mutator2.png")