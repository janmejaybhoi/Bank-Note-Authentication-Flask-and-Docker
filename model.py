# Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle

df = pd.read_csv("BankNote_Authentication.csv")

X = df.iloc[ :,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,shuffle=True)

# importing libaries for NN
from keras.layers import Dropout
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.optimizers import RMSprop


model = Sequential()
model.add(Dropout(0.1,input_shape =(4,)))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(50,activation= 'relu'))
model.add(Dropout(0.1))
model.add(tf.keras.layers.Flatten())
model.add(Dense(1, activation= 'sigmoid'))
optimizer = SGD(learning_rate=0.002)
model.compile(loss= 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
#SGD(learning_rate=0.002)

history = model.fit(X_train,y_train,epochs = 100, validation_data=(X_test, y_test))

import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()

plt.plot(epochs, loss, 'b', label='Training accuracy')
plt.plot(epochs, val_loss, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()



model.predict_classes([[-2.1674,0.12415,-1.0465,-0.86208],[0.80355,	2.8473,	4.3439,0.6017],[1.5356	,9.1772,	-2.2718	,-0.73535],[-0.64472,	-4.6062	,8.347,	-2.7099],
                       [-2.41	,3.7433	,-0.40215	,-1.2953],[-1.3066,	0.25244	,0.7623,	1.7758]])


# import pickle
# pickle_out = open("model.pkl","wb")
# pickle.dump(model,pickle_out)
# pickle_out.close()

model.save("model.h5")
print("Saved model to disk")