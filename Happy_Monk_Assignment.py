#!/usr/bin/env python
# coding: utf-8

#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

data = pd.read_csv('cancer data (1).csv') # Importing the dataset

data
data.info()
data.isnull().sum() 
data = data.drop(["id","Unnamed: 32"], axis="columns")   # removing the unwanted columns
data
data.info()

data["diagnosis"].unique()      

from sklearn import preprocessing          #applying lable-encoder to convert categorical variable into numerical.
Encode = preprocessing.LabelEncoder()
Encode.fit(['M','B'])
data["diagnosis"] = Encode.transform(data['diagnosis'])

data["diagnosis"].unique()

data

x = data.drop("diagnosis", axis= 1)
x
x = np.asanyarray(x)     # Independant variable
x

y = np.asanyarray(data["diagnosis"])    #Dependent veriable
y

# Split the dataset into training and testing sets
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=40)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# Define the model architecture
model = Sequential()
model.add(Dense(16, activation = "relu", input_dim = 30)) # inpte layer
model.add(Dense(32, activation = "relu"))                 #1st hiden layer
model.add(Dense(1, activation = "sigmoid"))               #output layer

# Compile the model
model.compile(optimizer="adam",            # optimizer adjest the model wieghts to maximize a loss fuctions.
             loss = "binary_crossentropy", # For integer target we using "sparse_categorical_crossentropy"
             metrics = ['accuracy'])       # For finalizing the model & make it comletely ready for use

history = model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=0, validation_data=(x_test, y_test))
history

# Evaluate the model on training and testing data
train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=0)
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

# Calculating F1-Score
y_train_pred = model.predict(x_train).round()
y_test_pred = model.predict(x_test).round()

f1_train = f1_score(y_train, y_train_pred)
f1_test = f1_score(y_test, y_test_pred)

# Print the evaluation metrics
print("Train Loss:", train_loss*100)
print("Test Loss:", test_loss*100)
print("Train Accuracy:", train_accuracy*100)
print("Test Accuracy:", test_accuracy*100)
print("Train F1-Score:", f1_train*100)
print("Test F1-Score:", f1_test*100)

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

#F1-Score for both Training and Testing
print("Train F1-Score:", f1_train*100)
print("Test F1-Score:", f1_test*100)
