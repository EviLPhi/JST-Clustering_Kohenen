#!/usr/bin/env python
# coding: utf-8

# # Kohenen Network

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as pit
import copy


# ## Input Studi_kasus_clustering_dg_Kohonen_Net.xlsx

# In[ ]:


data = pd.read_excel("Studi_kasus_clustering_dg_Kohonen_Net.xlsx",skiprows=3)


# In[3]:


train = data.copy()
train = train.drop(['No'],axis=1)
train = train.drop(['Cluster'],axis=1)


# In[4]:


train


# ### Deklarasi Kohenen

# In[5]:


learning_rate = 0.6
faktor_penurunan = 0.5
R = 0.5
W = np.array([[1,3],[5,1],[-1,0]])

def hitung(temp,bobot):
    hasil = 0
    for t in range(len(bobot)) :
        hasil += np.power(temp[t]-bobot[t],2)
    return hasil

def ubah_bobot(temp,index):
    for i in range(len(temp)):
        W[index][i] = round(W[index][i] + learning_rate*(temp[i]-W[index][i]),3)


# ## Proses Learning

# In[6]:


epoch = 1
stop = False
while not stop:
    S = []
    print("\nEpoch : ",epoch)
    for i in range(len(train['X'])):
        W_lama = copy.deepcopy(W)
        temp = np.array([train['X'][i],train['Y'][i]])
        D = []
        for x in range(len(W)):
            D.append(round(hitung(temp,W[x]),3))
        print("Input : ",temp,"\n\t --> D : ",D)
        ubah_bobot(temp,D.index(min(D)))
        print("\tBobot Baru : ",W)
        print("\tBobot Lama : ",W_lama)
        selisih = abs(W_lama-W)
        S.append(selisih)
    if(np.mean(S) < R):
            stop = True
    epoch += 1
    learning_rate = learning_rate*faktor_penurunan
print("STOP!!!! \nW = ",W)


# ## Proses Testing

# In[7]:


cluster = []
for i in range(len(train['X'])):
    temp = [train['X'][i],train['Y'][i]]
    D = []
    for x in range(len(W)):
        D.append(hitung(temp,W[x]))
    cluster.append(D.index(min(D)))
print("Cluster : ",cluster)


# ## Menyimpan Hasil Testing ke Data

# In[8]:


train['Cluster'] = cluster


# In[9]:


train


# In[10]:


data


# In[11]:


data["Hasil_Kohenen"] = cluster


# In[12]:


data.loc[data['Hasil_Kohenen']==0]


# In[13]:


data.loc[data['Hasil_Kohenen']==1]


# In[14]:


data.loc[data['Hasil_Kohenen']==2]


# ## Visualiasi Data

# In[15]:


sns.relplot(data=data, x ='X',y = 'Y', hue = 'Hasil_Kohenen',style="Cluster")

