#!/usr/bin/env python
# coding: utf-8

# # Kohenen Network

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as pit
import copy


# ## Input

# In[2]:


data = pd.read_excel("Studi_kasus_clustering_dg_Kohonen_Net.xlsx",skiprows=3)
train = data.copy()
train = train[0:0]


# In[3]:


train.columns = ['A','B','C','D']
train['A'] = [1,1,0,0]
train['B'] = [0,0,0,1]
train['C'] = [1,0,0,0]
train['D'] = [0,0,1,1]
train


# ### Deklarasi Kohenen

# In[4]:


# learning_rate = float(input("Learning Rate = "))
# faktor_penurunan = float(input("Faktor Penurunan = "))
learning_rate = 0.6
faktor_penurunan = 0.5
R = float(input("R = "))
T = float(input("Threshold = "))
# W = [[rd.random(),rd.random(),rd.random(),rd.random()],
#      [rd.random(),rd.random(),rd.random(),rd.random()]]

W = [[0.2, 0.6, 0.5, 0.9], [0.8, 0.4, 0.7, 0.3]]
W = np.array(W)
def hitung(temp,bobot):
    hasil = 0
    for t in range(len(bobot)) :
        hasil += np.power(temp[t]-bobot[t],2)
    return hasil

def ubah_bobot(temp,index):
    W[index] = W[index] + learning_rate*(temp-W[index])
#     for i in range(len(temp)):
#         W[index][i] = (W[index][i] + learning_rate*(temp[i]-W[index][i]))


# ## Proses Learning

# In[5]:


epoch = 1
stop = False
while not stop:
    S = []
    U = []
    print("\nEpoch : ",epoch)
    for x in train:
        W_lama = copy.deepcopy(W)
        temp = np.array(list(train[x]))
        D = []
        for x in range(len(W)):
            D.append(hitung(temp,W[x]))
        print("Input : ",temp,"\n\t --> D : ",D)
        ubah_bobot(temp,D.index(min(D)))
#         print("\tBobot Lama : ",W_lama)
#         print("\tBobot Baru : ",W)
#         selisih = abs(hitung(temp,W[D.index(min(D))]) - hitung(temp,W_lama[D.index(min(D))]))
        selisih = abs(W - W_lama)
        print("D minimal = ",min(D))
#         print("Selisih = ",selisih)
        U.append(selisih)
        S.append(min(D))
    if(np.max(S) < R) or (epoch >= 1000) or (np.max(U) < T):
        stop = True
    else : 
        epoch += 1
        learning_rate = learning_rate*faktor_penurunan
    print("\nJari-Jari : ",R,">",np.max(S),np.max(S) < R)
    print("Theta = ",T,">",np.max(U),np.max(U) < T)
print("STOP -> Epoch : ",epoch," \nW : \n",W,
     "\nRound(W) :\n ",np.around(W,decimals=1))


# In[6]:


cluster = []
for i in train:
    temp = train[i]
    D = []
    for x in range(len(W)):
        D.append(hitung(temp,W[x]))
    cluster.append(D.index(min(D)))
print("Cluster : ",cluster)


# ## Menyimpan Hasil Testing ke Data

# In[7]:


train = train.transpose()


# In[8]:


train['Cluster'] = cluster
train


# In[9]:


train.loc[train['Cluster']==0]


# In[10]:


train.loc[train['Cluster']==1]


# ## Visualiasi Data

# In[ ]:




