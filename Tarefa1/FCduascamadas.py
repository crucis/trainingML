import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
#import math
import numpy
import pandas
import time


#CONSTANTS

# fix random seed for reproducibility
SEED = 7
numpy.random.seed(SEED)
TRAININGPOINTSPERCENTAGE = 1
BATCHSIZE = 5
EPOCH = 200

########################

#load iris dataset
dataframe = pandas.read_csv("iris.csv",header=None)
dataset = dataframe.values
# split into training and validation data for input and output variables
#numberOfPoints = len(dataset)
#trainPoints = int(numberOfPoints*TRAININGPOINTSPERCENTAGE)
# training data
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]
# validation data
#X_test = dataset[trainPoints:,0:4].astype(float)
#Y_test = dataset[trainPoints:,4]


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
#encoder.fit(Y_test)
encoded_Y = encoder.transform(Y)
#encoded_Y_test = encoder.transform(Y_test)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
#dummy_y_test = np_utils.to_categorical(encoded_Y_test)

hidden_nodes = [10,100,1000,10000]
print("----------------------------------")
print("Using Batch =",BATCHSIZE ,"Epoch = ",EPOCH, "Seed = ",SEED)
for i in hidden_nodes:
# create model
	# time measure
	start_time = time.time()
	# Model
	print("----------------------------------")
	print("Creating model with hidden_nodes =",i)
	def baseline_model():
		# create model
		model = Sequential()
		model.add(Dense(i, input_dim=4, init='normal', activation='relu'))
		#model.add(Dense(10, init='normal', activation='relu'))
		model.add(Dense(3, init='normal', activation='softmax'))
		# Compile model
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		return model


# evaluate the model

	estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=EPOCH, batch_size=BATCHSIZE, verbose=0)

	kfold = KFold(n=len(X), n_folds=10, shuffle=True, random_state=SEED)
	results = cross_val_score(estimator,X,dummy_y,cv=kfold)
	print("Baseline:%.2f%%(%.2f%%)" %(results.mean()*100,results.std()*100))
	# Show time elapsed
	print("elapsed time = %.2f s"%(time.time()-start_time))

#The results are summarized as both the mean and standard deviation of the model accuracy on the dataset. This is a reasonable estimation of the performance of the model on unseen data. It is also within the realm of known top results for this problem.
