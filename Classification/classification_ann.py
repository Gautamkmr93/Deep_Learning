import pandas as pd
import io
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#data importing
data=pd.read_csv('data.csv')
#print(data.head(5))

#deleting extra columns
del data['Unnamed: 32']

print(data.head())

#data Preprocessing
import seaborn as sns
ax = sns.countplot(data['diagnosis'], label= 'Count')
B,M = data['diagnosis'].value_counts()
print('Benign', B)
print('Malignanat', M)


X = data.iloc[:, 2:].values
y = data.iloc[:, 1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


import keras
from keras.models import Sequential
from keras.layers import Dense


#adding the input and first hidden layer
classifier = Sequential()
#adding the 1st hidden layer
classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'sigmoid', input_dim = 30))
#adding the 2nd hidden layer
classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'sigmoid'))
#adding the 3rd hidden layer
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'sigmoid'))
#adding the 4th hidden layer
classifier.add(Dense(1,activation='sigmoid'))

classifier.compile(optimizer="Adam", loss='binary_crossentropy', metrics=['accuracy'])
#train your model
classifier.fit(X_train, y_train, batch_size=100, epochs=7)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

print(y_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm,annot=True)
plt.savefig('confusion_matrix.png')