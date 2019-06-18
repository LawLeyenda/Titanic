import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('TrainCLeaned.csv')

fig = plt.figure(figsize=(18,6))

plt.subplot2grid((2,3),(0,0))
df.Survived.value_counts(normalize=True).plot(kind="bar")
plt.title("Percent Survived")

plt.subplot2grid((2,3),(0,1))
plt.scatter(df.Survived, df.Age, alpha=.25)
plt.title("Age over survival")

plt.subplot2grid((2,3),(0,2))
df.Pclass.value_counts(normalize=True).plot(kind="bar")
plt.title("PClass")


plt.subplot2grid((2,3),(1,0), colspan=2)
for x in [1,2,3]:
    df.Age[df.Pclass == x].plot(kind="kde")
plt.title("Class over Age")
plt.legend(("1st", "2nd", "3rd"))

plt.subplot2grid((2,3),(1,2))
df.Embarked.value_counts(normalize=True).plot(kind="bar")
plt.title("Embarked")
plt.show()
