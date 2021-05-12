import tensorflow as tf
import pandas as pd
import numpy as np

data = pd.read_csv('gpascore.csv')
#print(data.isnull().sum()) #find null
#data = data.fillna(100) # fill blank
#print(data['gpa'].min())
#exit()
data = data.dropna() #delete blank
ydata = data['admit'].values
xdata = []

for i, rows in data.iterrows():
    xdata .append([ rows['gre'], rows['gpa'], rows['rank'] ])


model = tf.keras.models.Sequential([
    #number of hidden layers
    tf.keras.layers.Dense(64, activation = 'tanh'),
    tf.keras.layers.Dense(128,activation = 'tanh'),
    tf.keras.layers.Dense(1,activation = 'sigmoid') # 0 ~ 1
])

#binary_crossentropy : use for the probs of probabilitieswhen the result should be 0 ~ 1
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(np.array(xdata),np.array(ydata),epochs=1000) # run

#expectation
predict = model.predict([[750,3.70,3],[400,2.2,1]])
print(predict)