import numpy as np
import pandas as pd

Train = pd.read_excel(open('TheTitanicDataSet.xlsm', 'rb'),
                           sheet_name='Train')

TestCleaned = pd.read_excel(open('TheTitanicDataSet.xlsm', 'rb'),
                           sheet_name='TestCleaned')

Train.to_csv('TrainCleaned.csv', index = False)
TestCleaned.to_csv('TestCleaned.csv', index = False)