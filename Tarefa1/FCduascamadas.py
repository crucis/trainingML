#Ver overfitting: criar uma FC com duas camadas (Dense, Act, Dense, Act). Testar [10, 100, 1000, 10000] neuronios ocultos, treinando por algumas epochs. Para cada uma das 4 redes, plotar a trainAcc e validAcc de acordo com a epoch.

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
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
VALIDATION% = 0.10

########################

#load iris dataset
dataframe = pandas.read_csv("iris.csv",header=None)
dataset = dataframe.values
# training data
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

# Vector with hidden_nodes on second layer
hidden_nodes = [10,100,1000,10000]
print("----------------------------------")
print("Using Batch =",BATCHSIZE ,"Epoch = ",EPOCH, "Seed = ",SEED)

# Create len(hidden_nodes) neural networks
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
		# Camada com i neurônios
		model.add(Dense(i, input_dim=4, init='normal', activation='relu'))
		# Camada de saída, 3 categorias
		model.add(Dense(3, init='normal', activation='softmax'))
		# Compile model
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		return model


# evaluate the model
	#KerasClassifier class will pass on to the fit() function internally all arguments used to train the neural network. # of epochs, batch size and debugging (0 = off).
	estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=EPOCH, batch_size=BATCHSIZE, verbose=0)

	#cross-validation uses 100% of the data. At each iteration, some p% of data is trained on, and (100-p)% is tested on, and the average score is returned.
	kfold = KFold(n=len(X), n_folds=10, shuffle=True, random_state=SEED)
	results = cross_val_score(estimator,X,dummy_y,cv=kfold)
	#The results are summarized as both the mean and standard deviation of the model accuracy on the dataset. This is a reasonable estimation of the performance of the model on unseen data.
	print("Baseline:%.2f%%(%.2f%%)" %(results.mean()*100,results.std()*100))
	# Show time elapsed
	print("elapsed time = %.2f s"%(time.time()-start_time))


