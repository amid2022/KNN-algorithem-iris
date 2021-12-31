#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


iris = datasets.load_iris()


# In[3]:


iris.data.shape


# In[4]:


iris.feature_names
#iris.data


# In[5]:


iris.target_names


# In[6]:


iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df


# In[7]:


iris_df['target'] = iris.target
iris_df


# In[8]:


# Visual EDA
pd.plotting.scatter_matrix(iris_df, c=iris.target, figsize=[11, 11], s=150)
plt.show()


# # KNN : K-Nearest Neighbors

# In[10]:


from sklearn import datasets
iris = datasets.load_iris()
x = iris.data[:, [2, 3]] #only use petal length and width
y = iris.target
plt.scatter(x[:,0],x[:,1], c=y)
plt.show()


# # Fit Data to Model

# In[11]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6, metric='minkowski',p=2)
x = iris.data
y = iris.target
knn.fit(x, y)


# # Predict

# In[12]:


iris.data


# In[13]:


xx = np.array([[5, 3, 1, 0.2]])
yy = knn.predict(xx)
print(yy)


# # Train and test

# In[15]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=
0.3, random_state=42, stratify=iris.target)


# In[16]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_predict = knn.predict(x_test)
y_predict


# In[17]:


knn.score(x_test, y_test)


# # Over Fitting and Under Fitting Testing

# In[21]:


neighbors = np.arange(1, 30)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,k in enumerate(neighbors):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(x_train, y_train)
    train_accuracy[i] = knn_model.score(x_train, y_train)
    test_accuracy[i] = knn_model.score(x_test, y_test)
    
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


# In[ ]:




