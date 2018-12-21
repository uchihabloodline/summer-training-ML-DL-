
# coding: utf-8

# In[10]:


import numpy as np
from sklearn.datasets import load_diabetes
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# #functions to handle the dataset

# In[11]:


def dataset_normalise(data):
    m = np.mean(data)
    n = np.std(data)
    normalised = (data-m)/n
    return normalised


# ## getting the dataset from imported library

# In[12]:


def get_data():
    diabetes = load_diabetes()
    features = np.array(diabetes.data)
    label = np.array(diabetes.target)
    label = np.reshape(label,(-1,1))
    label = dataset_normalise(label)
    print(features.shape,label.shape)
    return features,label


# In[13]:


X,Y = get_data()


# ##LR Model using Keras

# In[20]:


model = Sequential([ Dense(1, input_shape = (10,), activation=None) ])

model.compile(optimizer = 'adam', loss = 'mse')


# In[46]:


hst = model.fit(X,Y, batch_size=10, epochs=250, shuffle=True,validation_split=0.1)


# In[47]:


plt.plot(training_model.history['loss'],'r')
plt.plot(training_model.history['val_loss'],'g')
plt.show()


# In[42]:


print(r2_score(y_pred = model.predict(X), y_true = Y))


# In[44]:


LR_ = LinearRegression()
LR_.fit(X,Y)


# In[45]:


print(LR_.score(X,Y))

