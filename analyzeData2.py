import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('TrainCLeaned.csv')

fig = plt.figure(figsize=(18,6))

plt.subplot2grid((3,4),(0,0))
df.Survived.value_counts(normalize=True).plot(kind="bar")
plt.title("Percent Survived")

plt.subplot2grid((3,4),(0,1))
df.Sex.value_counts(normalize=True).plot(kind="bar")
plt.title("Sex")

plt.subplot2grid((3,4),(0,2))
df.Sex[df.Survived == 1].value_counts(normalize=True).plot(kind="bar")
plt.title("Sex of Survived")

#plt.subplot2grid((3,4),(1,0), colspan=4)
#for x in [1,2,3]:
 #   df.Survived[df.Pclass == x].plot(kind="kde")
#plt.title("Survived over Class")
#plt.legend(("1st", "2nd", "3rd"))

plt.subplot2grid((3,4),(2,1))
df.Survived[(df.Sex == "male") & (df.Pclass == 3)].value_counts(normalize=True).plot(kind="bar")
plt.title("Poor surviving males ")


plt.subplot2grid((3,4),(1,3))
df.Survived[(df.Sex == "female") & (df.Pclass == 3)].value_counts(normalize=True).plot(kind="bar")
plt.title("Poor surviving women ")

plt.subplot2grid((3,4),(1,2))
df.Survived[(df.Sex == "female") & (df.Pclass == 1)].value_counts(normalize=True).plot(kind="bar")
plt.title("rich surviving women ")


plt.show()

