import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('test.csv')

df1 = pd.read_csv('gender_submission.csv')

df3 = pd.concat([df,df1], axis = 1 )
# remove duplicated column 'PassengerId'
df3 = df3.loc[:,~df3.columns.duplicated()]
