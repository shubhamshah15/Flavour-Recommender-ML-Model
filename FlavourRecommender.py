#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas
dt = pandas.read_csv('flavour.csv')
dt


# In[3]:


dt['Age'].max() - dt['Age'].min()


# In[4]:


len(dt['Age'])


# In[6]:


from sklearn.preprocessing import LabelEncoder
Enc = LabelEncoder()
Enc.fit(['Male', 'Female'])


# In[7]:


dt['Gender'] = Enc.transform(dt['Gender'])


# In[8]:


dt


# In[11]:


x = dt.drop(columns = ['Flavour'])
y = dt.drop(columns = ['Age', 'Gender'])


# In[12]:


from sklearn.tree import DecisionTreeClassifier
CModel = DecisionTreeClassifier()
CModel.fit(x,y)


# In[13]:


age = 18
gender = Enc.transform(['Male'])
CModel.predict([  [age, gender]  ])


# In[14]:


Enc.inverse_transform([1])


# In[15]:


def flav_pred():
    age = int(input('Age:'))
    gen = input('Gender:').capitalize()
    gender = Enc.transform([gen])
    flav = CModel.predict([[age, gender]])
    print('Recommended Flavour:',flav[0])


# In[18]:


flav_pred()

