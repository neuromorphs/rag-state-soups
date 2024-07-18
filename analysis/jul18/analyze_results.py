import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the results
df = pd.read_csv('./results/results.csv')
df["layer_condition"] = df["condition"].str.split("_").str[1]
df["ssm_condition"] = df["condition"].str.split("_").str[3].astype(float)
df["conv_condition"] = df["condition"].str.split("_").str[5].astype(float)
df["correct"] = df["correct"].astype(float)

pivot_df = df.pivot_table(index='ssm_condition', columns='conv_condition', values='correct', aggfunc=np.mean)
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_df, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title('Mean Correct by ssm_condition and conv_condition')
plt.xlabel('conv_condition')
plt.ylabel('ssm_condition')
plt.savefig("condition_matrix.png")
plt.close()

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 4), dpi=300)
for idx, col in enumerate("layer_condition ssm_condition conv_condition".split(" ")):
    x = df[[col, "correct"]].groupby(col).mean()
    s = df[[col, "correct"]].groupby(col).std()
    axs[0][idx].errorbar(x.index, x["correct"], s["correct"], label=col.split("_")[0])
    axs[0][idx].legend()
    axs[1][idx].plot(x.index, x["correct"], label=col.split("_")[0])
    axs[1][idx].legend()
plt.plot()
plt.tight_layout()
plt.savefig("analysis.png")
plt.close()
