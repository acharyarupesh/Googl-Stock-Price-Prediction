#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd


# In[37]:


data = pd.read_csv("Merged_data.csv")


# In[17]:


df.head()


# In[44]:


df1 = data[['Close','Open']].copy()


# In[47]:


def CAPM(df):
    df= df.pct_change(1)
    df = df.dropna(axis=0)
    X = df
    Y_CAPM=data['Returns'][1:]
    X=np.matrix(X)
    Y_CAPM=np.matrix(Y_CAPM)
    intercept_arr1 = [1]*len(df['Open'])
    XT1 = np.matrix(np.asarray([np.asarray(intercept_arr1),np.asarray(df['Open'])]) )
    XTT1 = np.matrix(np.transpose(XT1))
    mult_1 = XT1@XTT1
    B1 = np.matrix(((np.linalg.inv(mult_1))@XT1))
    Y_CAPM=Y_CAPM.reshape(668,1)
    beta=B1@Y_CAPM
    print("BETA values are : {}".format(beta))
    Y_CAPM_HAT = np.matrix(XTT1@beta)
    rmse1=np.asarray(Y_CAPM - Y_CAPM_HAT)
    rmse_CAPM=np.sqrt(np.mean(rmse1*rmse1))
    plt.figure(figsize=(20,10))
    plt.plot(Y_CAPM)
    plt.plot(Y_CAPM_HAT)
    return Y_CAPM_HAT,rmse_CAPM,Y_CAPM


# In[52]:


Y_CAPM_HAT,rmse_CAPM,Y_CAPM=CAPM(df1)
print('RMSE= ',rmse_CAPM)


# In[49]:


rmse_CAPM


# In[ ]:




