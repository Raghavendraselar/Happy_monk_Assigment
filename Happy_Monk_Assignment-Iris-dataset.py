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

data = pd.read_csv('Iris.csv') # Importing the dataset

data
data.info()
data.isnull().sum() 

data = data.drop(["Id"], axis="columns")   # removing the unwanted columns
data
data.info()

data["Species"].unique()      

from sklearn import preprocessing          #applying lable-encoder to convert categorical variable into numerical.
Encode = preprocessing.LabelEncoder()
Encode.fit(['Iris-setosa','Iris-versicolor','Iris-virginica'])
data["Species"] = Encode.transform(data['Species'])
data["Species"].unique()

data

x = data.drop("Species", axis= 1)
x = np.asanyarray(x)
x

y = np.asanyarray(data["Species"])    #Dependent veriable
y

from sklearn.preprocessing import normalize
x_nor = normalize(x,axis=0)
x_nor

# Split the dataset into training and testing sets
x_train,x_test,y_train,y_test = train_test_split(x_nor,y,test_size=0.2,random_state=50)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

from keras.utils import np_utils
y_train1 = np_utils.to_categorical(y_train,num_classes=3)
y_test1 = np_utils.to_categorical(y_test,num_classes=3)
print("shape of y_train",y_train1.shape)
print("shape of y_test",y_test1.shape)

# Define the model architecture
model = Sequential()
model.add(Dense(32, activation = "relu", input_dim = 4))    # inpte layer
model.add(Dense(64, activation = "relu"))                    #1st hiden layer
model.add(Dense(3, activation = "softmax"))                   #output layer

# Compile the model
model.compile(optimizer="adam",                     # optimizer adjest the model wieghts to maximize a loss fuctions.
             loss = "categorical_crossentropy", 
             metrics = ['accuracy'])                # For finalizing the model & make it comletely ready for use

history = model.fit(x_train, y_train1, batch_size=20, epochs=47, verbose=1, validation_data=(x_test, y_test1))
history

y_train_pred = model.predict(x_train).round()
y_test_pred = model.predict(x_test).round()

# calculating accuracy
from sklearn.metrics import accuracy_score
test_accuracy = accuracy_score(y_test1, y_test_pred)
train_accuracy = accuracy_score(y_train1, y_train_pred)

# Print the evaluation metrics
print("Train Loss:", train_loss*100)
print("Test Loss:", test_loss*100)
print("Train Accuracy:", train_accuracy*100)
print("Test Accuracy:", test_accuracy*100)

# Ploting the loss function
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(["train", "Validation"], loc="upper right")

# Ploting the accuracy function
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend(["train", "Validation"], loc="lower right")

print("Train Accuracy:", train_accuracy*100)
print("Test Accuracy:", test_accuracy*100)
