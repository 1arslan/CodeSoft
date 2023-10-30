#!/usr/bin/env python
# coding: utf-8

# # TASK 01

# In[35]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[36]:


titanic =  pd.read_csv(r"C:\Users\ARSLAN\Desktop\tested.csv")
titanic


# In[37]:


titanic.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
titanic['Age'].fillna(titanic['Age'].median(), inplace=True)
titanic['Fare'].fillna(titanic['Fare'].median(), inplace=True)
titanic['Embarked'].fillna(titanic['Embarked'].mode()[0], inplace=True)
titanic = pd.get_dummies(titanic, columns=['Sex', 'Embarked'], drop_first=True)


# In[38]:


titanic


# In[37]:


X = titanic.drop(['Survived'], axis=1)
y = titanic['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[38]:


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# In[39]:


y_pred = model.predict(X_test)


# In[40]:


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

classification_rep = classification_report(y_test, y_pred)
print('Classification Report:')
print(classification_rep)


# # TASK 04

# In[21]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[22]:


data = pd.read_csv(r"C:\Users\ARSLAN\Desktop\advertising.csv") 
data


# In[23]:


X = data.drop("Sales", axis=1) 
y = data["Sales"]  


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[25]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[26]:


y_pred = model.predict(X_test)


# In[27]:


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)


# In[28]:


print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared (R2) Score: {r2}")


# # TASK 03

# In[29]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[30]:


data = pd.read_csv(r"C:\Users\ARSLAN\Desktop\IRIS.csv") 
data


# In[31]:


X = data.drop("species", axis=1)
y = data["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[32]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
k = 3 
knn = KNeighborsClassifier(n_neighbors=k)


# In[33]:


knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=data["species"].unique())


# In[34]:


print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{confusion}")
print("Classification Report:\n", report)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




