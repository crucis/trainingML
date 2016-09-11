#Aprender regularizacao: estudar regularizacao (no livro ou na internet), principalmente regularizacao L1, L2 e tecnica Dropout. Com 2 camadas e 1000 neuronios ocultos, criar redes com L1 = 0.001, L2 = 0.001 e dropout = 0.2 (total de 3 redes, cada tecnica individual em uma).Plotar trainAcc e validAcc para cada, de acordo com a epoch. 
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.regularizers import l2
from keras.regularizers import l1
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy
import pandas
import time


########################
#CONSTANTS and VARIABLES
########################

# Graphs options
SHOW_GRAPHS = 1
SAVE_GRAPHS = 1
SHOW_LOSS = 1
SHOW_ACC = 1

# fix random seed for reproducibility
SEED = 7
numpy.random.seed(SEED)
TRAININGPOINTSPERCENTAGE = 1

# Training Options
BATCHSIZE = 32
EPOCH = 1000
VALIDATIONPERC = 0.15

# Model Options
l1Reg = 0.001
l2Reg = 0.001
dropout = 0.2
hidden_nodes = 1000 # Vector with hidden_nodes on second layer

########################
#FUNCTIONS
########################

# defining function to display and/or save graphs
def plotGraph (filename, nodes, vecTrain, vecTest, nameVec):
	# Plot	
	print("--Now plotting ",nameVec,"--")
	#Plot Loss
	plt.plot(vecTrain)
	plt.plot(vecTest)
	titlestr = nameVec+" for "+str(nodes)+" hidden nodes with "+filename
	plt.title(titlestr)
	plt.ylabel(nameVec)
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='lower right')
	if SAVE_GRAPHS == 1:
		figurestr = "results/"+filename+"_"+nameVec+".png"
		plt.savefig(figurestr)
	if SHOW_GRAPHS == 1:
		plt.show()
	plt.clf()

########################
#PROGRAM
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


# Create 3 neural networks, each with different regularization methods
for i in range (0,3):
	# time measure
	start_time = time.time()

	########################
	# Model
	print("----------------------------------")
	print("Creating model with hidden_nodes =",hidden_nodes)
	# create model

	model = Sequential()
	# layer with hidden_nodes nodes
	if i == 0:
		model.add(Dense(hidden_nodes, input_dim=4, init='normal', activation='relu', W_regularizer = l1(l1Reg)))
		regStr = "L1_Regularization="+str(l1Reg)
	elif i == 1:
		model.add(Dense(hidden_nodes, input_dim=4, init='normal', activation='relu', W_regularizer = l2(l2Reg)))
		regStr = "L2_Regularization="+str(l2Reg)
	else:
		model.add(Dense(hidden_nodes, input_dim=4, init='normal', activation='relu'))
		model.add(Dropout(dropout))
		regStr = "Dropout="+str(dropout)
	print("Added",regStr)
	# output layer
	model.add(Dense(3, init='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	# Fit model	
	history = model.fit(X,dummy_y,validation_split=VALIDATIONPERC,nb_epoch=EPOCH,batch_size=BATCHSIZE,verbose=0)
	print("val_acc = %.4f"%history.history["val_acc"][EPOCH-1]," acc = %.4f"%history.history["acc"][EPOCH-1])	
	print("val_loss = %.4f"%history.history["val_loss"][EPOCH-1]," loss = %.4f"%history.history["loss"][EPOCH-1])	
	# Show time elapsed
	print("Elapsed time = %.2f s"%(time.time()-start_time))
	# Storing history
	acc = numpy.array(history.history['acc'])
	val_acc = numpy.array(history.history['val_acc'])
	loss = numpy.array(history.history['loss'])
	val_loss = numpy.array(history.history['val_loss'])
	# Plotting
	if SHOW_ACC == 1:
		plotGraph(regStr,nodes=hidden_nodes, vecTrain=acc, vecTest=val_acc, nameVec="acc")
	if SHOW_LOSS == 1:
		plotGraph(regStr,nodes=hidden_nodes, vecTrain=loss, vecTest=val_loss, nameVec="loss")

print("----------------------------------")
# eof
