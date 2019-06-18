import pandas as pd
import numpy as np

df = pd.read_csv('TrainCLeaned.csv')

df["Hyp"] = 0
df.loc[df.Sex == "female", "Hyp"] = 1

df["Result"] = 0
df.loc[df.Survived == df["Hyp"], "Result"] = 1

print(df["Result"].value_counts(normalize=True))
