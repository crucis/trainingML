#Aprender a utilizar regularizacao: criar uma FC com 2 camadas, numero de neuronios ocultos a sua escolha. Utilizar a regularizacao que quiser (pode usar mais de uma ao mesmo tempo). Tentar maximizar valAcc com o dataset MNIST.
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
SHOW_GRAPHS = 0
SAVE_GRAPHS = 1
SHOW_LOSS = 1
SHOW_ACC = 1

# fix random seed for reproducibility
SEED = 7
numpy.random.seed(SEED)

# Training Options
BATCHSIZE = 32
EPOCH = 10000
VALIDATIONPERC = 0.15

# Model Options
l1Reg = 0.008 # 0.008>91%
l2Reg = 0.013 # works with 0.001
dropout = 0.1
hidden_nodes = [15] # Vector with hidden_nodes on second layer use [x1, x2, x3, ..., xn]

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


print("----------------------------------")
# eof
