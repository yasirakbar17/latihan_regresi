#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df1 = pd.read_csv(r'E:\data1.csv')


# In[3]:


df1.head()


# In[4]:


data2 = pd.read_csv(r'E:\data2.csv')


# In[5]:


data2.head(3)


# In[6]:


data2 = data2.fillna(0)


# In[7]:


data2.isna().sum()


# In[8]:


data2.drop('date', axis = 1, inplace = True)


# In[9]:


data2 = data2.groupby('user_id').mean()


# In[10]:


data2.head()


# In[11]:


data2.shape


# In[12]:


import seaborn as sns
sns.heatmap(data2.corr(), annot = True);


# In[13]:


data2 = data2[data2['Saham_AUM'] != 0]


# In[14]:


import matplotlib.pyplot as plt


# In[15]:


X = data2.drop('Saham_AUM', axis = 1)
y = data2['Saham_AUM']

fig, axes = plt.subplots(2,4, figsize = (15, 8))
for i, col in enumerate(X.columns):
    sns.scatterplot(x = col, y = 'Saham_AUM', data = data2, ax=axes[i//4, i%4])
plt.tight_layout()
plt.show()


# In[16]:


data = data2[['Saham_AUM','Saham_invested_amount']]
data.head()


# In[17]:


data_sampel = data.sample(n = 140, random_state = 42)
data_sampel.head()


# In[18]:


data_sampel.reset_index(drop = True, inplace = True)


# In[19]:


def bulat(nilai):
    return int(nilai)
data_sampel.applymap(bulat)


# In[20]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[21]:


x = data_sampel['Saham_invested_amount']
y = data_sampel['Saham_AUM']
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 22)


# In[22]:


import numpy as np
X_train = np.array(X_train).reshape(-1,1)
X_test = np.array(X_test).reshape(-1,1)
y_train = np.array(y_train).reshape(-1,1)
y_test = np.array(X_test).reshape(-1,1)


# In[23]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[24]:


from sklearn.metrics import r2_score
r2_score(y_test, model.predict(X_test))


# In[25]:


plt.plot(model.predict(X_test), c = 'red', label = 'prediksi')
plt.plot(y_test, c = 'blue', label = 'y_test')
plt.title('Plot line prediksi & y_test')
plt.legend()
plt.show()


# In[ ]:




