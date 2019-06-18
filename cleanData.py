import pandas as pd
import numpy as np
import math

df = pd.read_csv('TrainCleaned.csv')

df["Age"] = df['Age'].fillna(df["Age"].dropna().median())

df["Fare"] = df['Fare'].fillna(df["Fare"].dropna().median())

df.loc[df["Sex"] == "male", "Sex"] = 0
df.loc[df["Sex"] == "female", "Sex"] = 1



df["Embarked"] = df["Embarked"].fillna("S")
df.loc[df["Embarked"] == "S", "Embarked"] = 0
df.loc[df["Embarked"] == "C", "Embarked"] = 1
df.loc[df["Embarked"] == "Q", "Embarked"] = 2

df.to_csv('TrainedCleaned.csv', index = False )



