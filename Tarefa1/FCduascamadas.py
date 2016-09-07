import keras
from keras.models import Sequential
from keras.layers import Dense
#import math
import numpy

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#load pima indians dataset for training
dataset = numpy.loadtxt("./dataset/iris.csv",delimiter=",")
# split into training and validation data for input and output variables
numberOfPoints = len(dataset)
trainPoints = int(numberOfPoints*0.7)
# training data
X = dataset[:trainPoints,0:8]
Y = dataset[:trainPoints,8]
# validation data
X_test = dataset[trainPoints:,0:8]
Y_test = dataset[trainPoints:,8]



# create model
model = Sequential()
model.add(Dense(12,input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1,init='uniform', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# Fit the model
model.fit(X,Y, nb_epoch=150, batch_size=128, validation_data=(X_test,Y_test))

# evaluate the model
scores = model.evaluate(X, Y)
scores2 = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1],scores[1]*100))
#print("\nval-acc: %.2f%%" % scores2[1]*100)


