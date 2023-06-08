#!/usr/bin/env python
# coding: utf-8

# In[1]:


#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[2]:


data = pd.read_csv('Iris.csv') # Importing the dataset


# In[3]:


data


# In[4]:


data.info()


# In[5]:


data.isnull().sum() 


# In[6]:


data = data.drop(["Id"], axis="columns")   # removing the unwanted columns


# In[7]:


data


# In[8]:


data.info()


# In[9]:


data["Species"].unique()      


# In[10]:


from sklearn import preprocessing          #applying lable-encoder to convert categorical variable into numerical.
Encode = preprocessing.LabelEncoder()
Encode.fit(['Iris-setosa','Iris-versicolor','Iris-virginica'])
data["Species"] = Encode.transform(data['Species'])


# In[12]:


data["Species"].unique()


# In[13]:


data


# In[15]:


x = data.drop("Species", axis= 1)
x = np.asanyarray(x)
x


# In[88]:


y = np.asanyarray(data["Species"])    #Dependent veriable
y


# In[99]:


from sklearn.preprocessing import normalize
x_nor = normalize(x,axis=0)
x_nor


# In[118]:


# Split the dataset into training and testing sets
x_train,x_test,y_train,y_test = train_test_split(x_nor,y,test_size=0.2,random_state=50)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# In[121]:


from keras.utils import np_utils
y_train1 = np_utils.to_categorical(y_train,num_classes=3)
y_test1 = np_utils.to_categorical(y_test,num_classes=3)


# In[130]:


print("shape of y_train",y_train1.shape)
print("shape of y_test",y_test1.shape)


# In[165]:


# Define the model architecture
model = Sequential()
model.add(Dense(32, activation = "relu", input_dim = 4))    # inpte layer
model.add(Dense(64, activation = "relu"))                    #1st hiden layer
model.add(Dense(3, activation = "softmax"))                   #output layer

# Compile the model
model.compile(optimizer="adam",                     # optimizer adjest the model wieghts to maximize a loss fuctions.
             loss = "categorical_crossentropy", 
             metrics = ['accuracy'])                # For finalizing the model & make it comletely ready for use


# In[174]:


history = model.fit(x_train, y_train1, batch_size=20, epochs=47, verbose=1, validation_data=(x_test, y_test1))
history


# In[175]:


# Calculating F1-Score
y_train_pred = model.predict(x_train).round()
y_test_pred = model.predict(x_test).round()


# In[188]:


from sklearn.metrics import accuracy_score
test_accuracy = accuracy_score(y_test1, y_test_pred)
train_accuracy = accuracy_score(y_train1, y_train_pred)

# Print the evaluation metrics
print("Train Loss:", train_loss*100)
print("Test Loss:", test_loss*100)
print("Train Accuracy:", train_accuracy*100)
print("Test Accuracy:", test_accuracy*100)


# In[189]:


# Ploting the loss function
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(["train", "Validation"], loc="upper right")


# In[190]:


# Ploting the accuracy function
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend(["train", "Validation"], loc="lower right")


# In[192]:


print("Train Accuracy:", train_accuracy*100)
print("Test Accuracy:", test_accuracy*100)

