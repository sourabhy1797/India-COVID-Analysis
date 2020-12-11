import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data4 = pd.read_csv(r'C:\Users\yadavs\Desktop\coorelation india\Datasets\niftybank.csv')
data4.index = data4['Date']
del data4['Date']
plt.plot(data4['corona'],data4['Open'])
plt.xlabel('No of Corona Patients Registered')
plt.ylabel('Open of NIFTY Bank')
corr = data4.corr()
corr