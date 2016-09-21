#Aprender a utilizar regularizacao: criar uma FC com 2 camadas, numero de neuronios ocultos a sua escolha. Utilizar a regularizacao que quiser (pode usar mais de uma ao mesmo tempo). Tentar maximizar valAcc com o dataset MNIST.
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.datasets import mnist
from keras.regularizers import l2
from keras.regularizers import l1
#from sklearn.preprocessing import LabelEncoder
#from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy
#import pandas
import time
from datetime import timedelta


########################
#CONSTANTS and VARIABLES
########################

# Graphs options
SHOW_GRAPHS = 0
SAVE_GRAPHS = 1
SHOW_LOSS = 1
SHOW_ACC = 1
SAVE_CSV = 0

# fix random seed for reproducibility
SEED = 7
numpy.random.seed(SEED)

# Training Options
BATCHSIZE = 200
EPOCH = 25


# Model Options
l1Reg = [0]#[0,0.0001,0.0005,0.001,0.005] # 
l2Reg = [0]#[0,0.0001,0.0005,0.001,0.005] # 
dropout = [0]#[0.2]
hidden_nodes = [784] #[200, 500, 784] 

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
	plt.legend(['train', 'test'], loc='best')
	strAnnotation = "val_"+nameVec+"="+str(vecTest[len(vecTest)-1])
	plt.text(2,min(vecTrain),strAnnotation, fontsize=14)
	if SAVE_GRAPHS == 1:
		output_dir="results/"+str(nodes)+"nodes/"+str(EPOCH)+"Epoch"
		mkdir_p(output_dir) # Verifies if directory exists, and creates it if necessary
		figurestr = output_dir+"/"+filename+"_"+nameVec+".png"
		plt.savefig(figurestr)
	if SHOW_GRAPHS == 1:
		plt.show()
	plt.clf()

########################
#PROGRAM
########################

# load mnist dataset
(X,Y),(X_test, Y_test) = mnist.load_data()
# flatten 28*28 images to a 784 vector for each image
num_pixels = X.shape[1]*X.shape[2]
X = X.reshape(X.shape[0],num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0],num_pixels).astype('float32')
# Data Preprocessing
#X -= numpy.mean(X)
#X_test -= numpy.mean(X)
X /= 255
X_test /= 255
# hot encode outputs
Y = np_utils.to_categorical(Y)
Y_test = np_utils.to_categorical(Y_test)
num_classes = Y_test.shape[1]

print("----------------------------------")
print("Using Batch =",BATCHSIZE ,"Epoch = ",EPOCH)

#define csv variable
if SAVE_CSV == 1:
	csvArray = numpy.array(["hidden_nodes","l2Reg","dropout","val_acc","acc","val_loss","loss"])


# define model
for h in range(0, len(hidden_nodes)):
	for j in range(0,len(dropout)):	
		for i in range(0,len(l1Reg)):
			# time measure

			start_time = time.time()
			print("----------------------------------")
			print("Creating model with hidden_nodes =",hidden_nodes[h])
			# create model
			model = Sequential()
			# creates layer with hidden_nodes nodes and chooses which regularization it will use
			model.add(Dense(hidden_nodes[h], input_dim=num_pixels, init='he_normal', activation='relu'))#, W_regularizer = l2(l2Reg[i])))
			regStr = "No L1-L2 Regularization"#+str(l2Reg[i])#+" L2_Regularization="+str(l2Reg)
			#model.add(Dropout(dropout[j]))
			regStr = regStr+"No Dropout"#%dropout[j]
			# output layer
			model.add(Dense(num_classes, init='he_normal', activation='softmax'))
			print("Added",regStr)
			# Compile model
			model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
			history = model.fit(X,Y,validation_data=(X_test,Y_test),nb_epoch=EPOCH,batch_size=BATCHSIZE,verbose=2)
			# Show time elapsed
			print("Elapsed time = ",str(timedelta(seconds=(time.time()-start_time))))
			# Storing history
			acc = numpy.array(history.history['acc'])
			val_acc = numpy.array(history.history['val_acc'])
			loss = numpy.array(history.history['loss'])
			val_loss = numpy.array(history.history['val_loss'])
			# Storing in csvArray
			if SAVE_CSV:
				a = numpy.array([hidden_nodes[h],l1Reg[i],dropout[j],val_acc[EPOCH-1],acc[EPOCH-1],val_loss[EPOCH-1],loss[EPOCH-1]])
				csvArray = numpy.vstack([csvArray,a])
			# Plotting
			if SHOW_ACC == 1:
				plotGraph(regStr,nodes=hidden_nodes[h], vecTrain=acc, vecTest=val_acc, nameVec="acc")
			if SHOW_LOSS == 1:
				plotGraph(regStr,nodes=hidden_nodes[h], vecTrain=loss, vecTest=val_loss, nameVec="loss")

if SAVE_CSV:
	print("Saving results to test.csv")
	numpy.savetxt("test.csv",csvArray,delimiter=",",fmt="%s")

print("----------------------------------")
# eof
