#Ver underfitting (breve): criar uma FC com 1000 neuronios ocultos. Para as tecnicas L1, L2 e Dropout, aumentar a regularizacao ate gerar underfitting. Gerar grafico eh opcional.

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
dropout = 0.4
hidden_nodes = [1000] # Vector with hidden_nodes on second layer use [x1, x2, x3, ..., xn]

########################
#FUNCTIONS
########################

def mkdir_p(mypath):
    #Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

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
		output_dir="results/"+str(nodes)+"nodes"
		mkdir_p(output_dir) # Verifies if directory exists, and creates it if necessary
		figurestr = output_dir+"/"+filename+"_"+nameVec+".png"
		plt.savefig(figurestr)
	if SHOW_GRAPHS == 1:
		plt.show()
	plt.clf()

########################
#PROGRAM
########################

#load iris dataset
dataframe = pandas.read_csv("dataset/iris.csv",header=None)
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
for h in range(0, len(hidden_nodes)):
	for i in range (0,3):
		# time measure
		start_time = time.time()

		########################
		# Model
		print("----------------------------------")
		print("Creating model with hidden_nodes =",hidden_nodes[h])
		# create model

		model = Sequential()
		# creates layer with hidden_nodes nodes and chooses which regularization it will use
		if i == 0:
			model.add(Dense(hidden_nodes[h], input_dim=4, init='normal', activation='relu', W_regularizer = l1(l1Reg)))
			# output layer
			model.add(Dense(3, init='normal', activation='softmax', W_regularizer = l1(l1Reg)))
			regStr = "L1_Regularization="+str(l1Reg)
		elif i == 1:
			model.add(Dense(hidden_nodes[h], input_dim=4, init='normal', activation='relu', W_regularizer = l2(l2Reg)))
			regStr = "L2_Regularization="+str(l2Reg)
			# output layer
			model.add(Dense(3, init='normal', activation='softmax', W_regularizer = l2(l2Reg)))
		else:
			model.add(Dense(hidden_nodes[h], input_dim=4, init='normal', activation='relu'))
			model.add(Dropout(dropout))
			# output layer
			model.add(Dense(3, init='normal', activation='softmax'))
			regStr = "Dropout="+str(dropout)
		print("Added",regStr)

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
			plotGraph(regStr,nodes=hidden_nodes[h], vecTrain=acc, vecTest=val_acc, nameVec="acc")
		if SHOW_LOSS == 1:
			plotGraph(regStr,nodes=hidden_nodes[h], vecTrain=loss, vecTest=val_loss, nameVec="loss")

print("----------------------------------")
# eof
