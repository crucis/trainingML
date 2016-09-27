from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.utils import np_utils
from keras.datasets import cifar10
#from keras.regularizers import l2, l1
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, Deconvolution2D
from keras import backend as K
from random import uniform
import matplotlib.pyplot as plt
import numpy
#import math
import sys



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
BATCHSIZE = 128
EPOCH = 1


# Model Options
kernel1 = [128,3] # 32 Kernels = 2x2
kernel2 = [3,3]
kernel3 = [64,3]
dropout = [0.05, 0.1]
pooling = 2



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
def converter(a,b):
	for i in range(0,b.shape[0]):
		a[i,0,:,:] = rgb2gray(b[i,:,:,:].transpose(1,2,0))

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

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("results/Test4/logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    



########################
#PROGRAM
########################

#### load cifar10 dataset
(Y,labels),(Y_test, labels_test) = cifar10.load_data()

Y = Y[(labels == 8)[:,0]]
Y_test = Y_test[(labels_test == 8)[:,0]]

X = numpy.zeros((Y.shape[0],1,Y.shape[2],Y.shape[3]))
X_test = numpy.zeros((Y_test.shape[0],1,Y.shape[2],Y.shape[3]))

# Convert RGB to grayscale to create our input
converter(X,Y)
converter(X_test,Y_test)

#### Data Preprocessing
# convert inputs and outputs to float32
X = X.astype('float32')
Y = Y.astype('float32')
X_test = X_test.astype('float32')
Y_test = Y_test.astype('float32')
# normalize inputs and outputs from 0-255 to 0.0-1.0
X /= 255
Y /= 255
X_test /= 255
Y_test /= 255

nImages = 2048

#G = X[:nImages]
#F = Y[:nImages]
#G_test = X_test[:math.ceil(nImages/5)]
#F_test = Y_test[:math.ceil(nImages/5)]
G = X
F = Y
G_test = X_test
F_test = Y_test

#### Model
print("----------------------------------")
print("Using Batch =",BATCHSIZE ,"Epoch = ",EPOCH)
model = Sequential()
# Layers
model.add(Convolution2D(8, kernel1[1], kernel1[1], border_mode='same', init='he_normal', input_shape=(1, 32, 32), activation='relu'))
model.add(Convolution2D(16, kernel1[1], kernel1[1], border_mode='same', init='he_normal', activation='relu'))
model.add(Convolution2D(32, kernel1[1], kernel1[1], border_mode='same', init='he_normal', activation='relu'))
model.add(Convolution2D(64, kernel1[1], kernel1[1], border_mode='same', init='he_normal', activation='relu'))
model.add(Convolution2D(32, kernel1[1], kernel1[1], border_mode='same', init='he_normal', activation='relu'))
model.add(Convolution2D(16, kernel1[1], kernel1[1], border_mode='same', init='he_normal', activation='relu'))
model.add(Convolution2D(8, kernel1[1], kernel1[1], border_mode='same', init='he_normal', activation='relu'))
model.add(Convolution2D(3 , kernel2[1], kernel2[1], border_mode='same', init='he_normal', activation='relu'))
model.add(Lambda(lambda x: K.clip(x, 0.0, 1.0)))

# Compile
model.compile(loss='mean_squared_error', optimizer='nadam', metrics=['mean_squared_error'])


# Fit
history = model.fit(G,F,validation_data=(G_test,F_test),nb_epoch=EPOCH,batch_size=BATCHSIZE,verbose=1)

#### Show results
MSE = numpy.array(history.history['mean_squared_error'])
val_MSE = numpy.array(history.history['val_mean_squared_error'])
loss = numpy.array(history.history['loss'])
val_loss = numpy.array(history.history['val_loss'])
# Plotting
if SHOW_ACC == 1:
	plotGraph('ColorizingTest1',nodes=1, vecTrain=MSE, vecTest=val_MSE, nameVec="mean_squared_error")
if SHOW_LOSS == 1:
	plotGraph('ColorizingTest1',nodes=1, vecTrain=loss, vecTest=val_loss, nameVec="loss")

print('MSE =',MSE[EPOCH-1],' val_MSE =',val_MSE[EPOCH-1],' loss =',loss[EPOCH-1],' val_loss =',loss[EPOCH-1])


# Getting a result sample for model
pred = model.predict(G)

for i in range(G.shape[0]*0.2):
	_,((ax1,ax2),(ax3,_)) = plt.subplots(2,2,sharey='row',sharex='col')
	
	n = uniform(0,G.shape[0])

	ax1.imshow(G[n,0,:,:],cmap='gray')
	ax1.set_title('Input_%s'%i)

	l = pred[n].transpose(1,2,0)
	ax2.imshow(l)
	ax2.set_title('Output_%s'%i)
	
	ax3.imshow(F[n].transpose(1,2,0))
	ax3.set_title('Original_%s'%i)

	titlestr = 'Epochs='+str(EPOCH)+'MSE=%.4f'%MSE[EPOCH-1]
	plt.title(titlestr)
	plt.grid(b=False)

	outDir = 'results/Test4/'
	mkdir_p(outDir)
	plt.savefig(outDir+'sample_%s.png'%i)
#	plt.show()
	plt.clf()
sys.stdout = Logger()
print(model.summary())


print("----------------------------------")
# eof
