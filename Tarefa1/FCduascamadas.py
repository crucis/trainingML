#Ver overfitting: criar uma FC com duas camadas (Dense, Act, Dense, Act). Testar [10, 100, 1000, 10000] neuronios ocultos, treinando por algumas epochs. Para cada uma das 4 redes, plotar a trainAcc e validAcc de acordo com a epoch.
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy
import pandas
import time


#CONSTANTS and VARIABLES

# fix random seed for reproducibility
SEED = 7
numpy.random.seed(SEED)
TRAININGPOINTSPERCENTAGE = 1
BATCHSIZE = 32
EPOCH = 1000
VALIDATIONPERC = 0.15

hidden_nodes = [10,100,1000,10000] # Vector with hidden_nodes on second layer
j = 0 # Used to count


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


print("----------------------------------")
print("Using Batch =",BATCHSIZE ,"Epoch = ",EPOCH, "Validation_split = ",VALIDATIONPERC)


# Create len(hidden_nodes) neural networks
for i in hidden_nodes:
	# time measure
	start_time = time.time()
	# Model
	print("----------------------------------")
	print("Creating model with hidden_nodes =",i)
	# create model
	def baseline_model():
		model = Sequential()
		# layer with i nodes
		model.add(Dense(i, input_dim=4, init='normal', activation='relu'))
		# output layer
		model.add(Dense(3, init='normal', activation='softmax'))
		# Compile model
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		return model
	# Fit model	
	history = baseline_model().fit(X,dummy_y,validation_split=VALIDATIONPERC,nb_epoch=EPOCH,batch_size=BATCHSIZE,verbose=0)
	print("val_acc = %.4f"%history.history["val_acc"][EPOCH-1]," acc = %.4f"%history.history["acc"][EPOCH-1])		
	# Show time elapsed
	print("Elapsed time = %.2f s"%(time.time()-start_time))
	# Storing history
	if (j == 0):
		loss = numpy.array(history.history['loss'])
		val_loss = numpy.array(history.history['val_loss'])
	else:
		loss = numpy.vstack([loss,history.history['loss']])
		val_loss = numpy.vstack([val_loss,history.history['val_loss']])
	j+=1
# Plot	
print("----------------------------------")
print("Now plotting")
for h in range(0, len(hidden_nodes)):
	plt.plot(loss[h,:])
	plt.plot(val_loss[h,:])
	titlestr = "model loss for "+str(hidden_nodes[h])+" hidden nodes"
	plt.title(titlestr)
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='lower right')
	plt.show()

# eof
