# Primeira versão da rede neural capaz de colorir fotos preto e brancas.

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.regularizers import l2
from keras.regularizers import l1
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
import matplotlib.pyplot as plt
import numpy
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
EPOCH = 10


# Model Options
kernel1 = [16,3] # 32 Kernels = 2x2
kernel2 = [32,3]
kernel3 = [64,3]
dropout = [0.05, 0.1]
pooling = 2
l1Reg = 0.001
hidden_nodes = [288] #[200, 500, 784] 

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

def rgb2gray(rgb):
    return numpy.dot(rgb[...,:3], [0.299, 0.587, 0.114])

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

# load cifar10 dataset
(Y,a),(Y_test, a_test) = cifar10.load_data()
X = numpy.zeros((Y.shape[0],1,Y.shape[2],Y.shape[3]))
X_test = numpy.zeros((Y_test.shape[0],1,Y.shape[2],Y.shape[3]))

# Convert RGB to grayscale to create our input
for i in range(0, Y.shape[0]):
	X[i,0,:,:] = rgb2gray(Y[i,:,:,:].transpose(1,2,0))
for i in range(0,Y_test.shape[0]):
	X_test[i,0,:,:] = rgb2gray(Y_test[i,:,:,:].transpose(1,2,0))



if SAVE_CSV:
	print("Saving results to test.csv")
	numpy.savetxt("test.csv",csvArray,delimiter=",",fmt="%s")

print("----------------------------------")
# eof
