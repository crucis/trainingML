
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
BATCHSIZE = 64
EPOCH = 250


# Model Options
kernel1 = [32,3] # 32 Kernels = 2x2
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

########################
#PROGRAM
########################

#### load cifar10 dataset
(Y,_),(Y_test, _) = cifar10.load_data()

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



nImages = 512



# Limit size of dataset
F = numpy.zeros((nImages,Y.shape[1],Y.shape[2],Y.shape[3]))
G = numpy.zeros((nImages,X.shape[1],X.shape[2],X.shape[3]))
F_test = numpy.zeros((128,Y_test.shape[1],Y_test.shape[2],Y_test.shape[3]))
G_test = numpy.zeros((128,X_test.shape[1],X_test.shape[2],X_test.shape[3]))
for i in range(0,F.shape[0]):
	F[i] = Y[i]
for i in range(0,G.shape[0]):
	G[i] = X[i]
for i in range(0,F_test.shape[0]):
	F_test[i] = Y_test[i]
for i in range(0,G_test.shape[0]):
	G_test[i] = X_test[i]

#### Model
print("----------------------------------")
print("Using Batch =",BATCHSIZE ,"Epoch = ",EPOCH)
model = Sequential()
# Layers
model.add(Convolution2D(kernel1[0], kernel1[1], kernel1[1], border_mode='same', init='he_normal', input_shape=(1, 32, 32), activation='relu'))
model.add(Convolution2D(kernel2[0], kernel2[1], kernel2[1], border_mode='same'))

# Compile
model.compile(loss='mean_squared_error', optimizer='nadam', metrics=['mean_squared_error'])
print(model.summary())

# Fit
history = model.fit(G,F,validation_data=(G_test,F_test),nb_epoch=EPOCH,batch_size=BATCHSIZE,verbose=1)

#### Show results
acc = numpy.array(history.history['mean_squared_error'])
val_acc = numpy.array(history.history['val_mean_squared_error'])
loss = numpy.array(history.history['loss'])
val_loss = numpy.array(history.history['val_loss'])
# Plotting
if SHOW_ACC == 1:
	plotGraph('ColorizingTest1',nodes=1, vecTrain=acc, vecTest=val_acc, nameVec="mean_squred_error")
if SHOW_LOSS == 1:
	plotGraph('ColorizingTest1',nodes=1, vecTrain=loss, vecTest=val_loss, nameVec="loss")

print('MSE =',acc[EPOCH-1],' val_MSE =',val_acc[EPOCH-1],' loss =',loss[EPOCH-1],' val_loss =',loss[EPOCH-1])

# Getting a result sample for model
pred = model.predict(G)
plt.imshow(G[120,0,:,:],cmap='gray')
mkdir_p('results/Test2')
plt.savefig('results/Test2/input.png')
plt.show()

for i in range(128):
    print (i)
    l = pred[i].transpose(1,2,0)
    plt.imshow(l)
    titlestr = 'Epochs='+str(EPOCH)
    plt.title(titlestr)
    plt.savefig('results/Test2/output.png')
    plt.show()

if SAVE_CSV:
	print("Saving results to test.csv")
	numpy.savetxt("test.csv",csvArray,delimiter=",",fmt="%s")

print("----------------------------------")
# eof
