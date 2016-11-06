from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.layers.normalization import BatchNormalization
#from keras.regularizers import l2, l1
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, Deconvolution2D
from keras import backend as K
from random import uniform
import matplotlib.pyplot as plt
import numpy
import math
import sys
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


# fix random seed for reproducibility
SEED = 7
numpy.random.seed(SEED)

# Training Options
BATCH_SIZE = 256
EPOCH = 50
nImages = pow(2,15)

# Model Options
folder = "Test8"
outDir = 'results/'+folder


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
def plotGraph (filename, vecTrain, vecTest, nameVec):
	# Plot	
	print("--Now plotting ",nameVec,"--")
	#Plot Loss
	plt.plot(vecTrain)
	plt.plot(vecTest)
	filename= filename
	titlestr = "Graphs for "+filename
	plt.title(titlestr)
	plt.ylabel(nameVec)
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='best')
	strAnnotation = "val_"+nameVec+"="+str(vecTest[len(vecTest)-1])
	plt.text(2,min(vecTrain),strAnnotation, fontsize=14)
	if SAVE_GRAPHS == 1:
		output_dir=outDir+"/graphs/"
		mkdir_p(output_dir) # Verifies if directory exists, and creates it if necessary
		figurestr = output_dir+filename+"_"+nameVec+".png"
		plt.savefig(figurestr)
	if SHOW_GRAPHS == 1:
		plt.show()
	plt.clf()

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(outDir+"/logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

def save3images(inp,out,original,folder):
	for i in range(math.floor(original.shape[0]*0.02)):
		_,((ax1,ax2),(ax3,_)) = plt.subplots(2,2,sharey='row',sharex='col')

		n = math.floor(uniform(0,original.shape[0]))

		ax1.imshow(inp[n,0,:,:],cmap='gray')
		ax1.set_title('Input_%s'%i)

		ax2.imshow(out[n].transpose(1,2,0))
		ax2.set_title('Output_%s'%i)

		ax3.imshow(original[n].transpose(1,2,0))
		ax3.set_title('Original_%s'%i)

		titlestr = 'Epochs='+str(epoch)+' BATCH_SIZE='+str(BATCH_SIZE)
		plt.title(titlestr)
		plt.grid(b=False)
		
		mkdir_p(outDir+'/samples/epoch'+str(folder))

		plt.savefig(outDir+'/samples/epoch'+str(folder)+'/sample_%s.png'%i)
		plt.clf()

########################
#PROGRAM
########################

sys.stdout = Logger()
# Create folder for tests
mkdir_p(outDir)

#### load cifar10 dataset
(Y,labels),(Y_test, labels_test) = cifar10.load_data()

# Choosing only boats
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



G = X[:nImages]
F = Y[:nImages]
G_test = X_test[nImages:nImages+math.ceil(nImages/5)]
F_test = Y_test[nImages:nImages+math.ceil(nImages/5)]


#### Models
print("----------------------------------")
# GENERATOR
def generator_model():
	model = Sequential()
	# Layers
	model.add(Convolution2D(8, 3, 3, border_mode='same', init='he_normal', input_shape=(1, 32, 32), activation='relu'))
	model.add(Convolution2D(16, 3, 3, border_mode='same', init='he_normal', activation='relu'))
	model.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal', activation='relu'))
	model.add(BatchNormalization())
	model.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal', activation='relu'))
	model.add(BatchNormalization())	
	model.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal', activation='relu'))
	model.add(BatchNormalization())
	model.add(Convolution2D(16, 3, 3, border_mode='same', init='he_normal', activation='relu'))
	model.add(Convolution2D(8, 3, 3, border_mode='same', init='he_normal', activation='relu'))
	model.add(Convolution2D(3 , 3, 3, border_mode='same', init='he_normal', activation='relu'))
	model.add(Lambda(lambda x: K.clip(x, 0.0, 1.0)))
	return model

# DISCRIMINATOR
def discriminator_model():
	model = Sequential()
	model.add(Convolution2D(8,3,3,border_mode='same',init='he_normal',input_shape=(3,32,32),activation='linear'))
	model.add(LeakyReLU(alpha=.2)) 
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Convolution2D(16,3,3,border_mode='same',init='he_normal',activation='linear'))
	model.add(LeakyReLU(alpha=.2)) 
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Convolution2D(32,3,3,border_mode='same',init='he_normal',activation='linear'))
	model.add(LeakyReLU(alpha=.2)) 
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Convolution2D(64,3,3,border_mode='same',init='he_normal',activation='linear'))
	model.add(LeakyReLU(alpha=.2)) 
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Flatten())
	model.add(Dense(512,init='he_normal',activation='linear'))
	model.add(LeakyReLU(alpha=.2))
	model.add(Dropout(0.2))
	model.add(Dense(256,init='he_normal',activation='linear'))
	model.add(LeakyReLU(alpha=.2))
	model.add(Dropout(0.2))
	model.add(Dense(1,init='he_normal',activation='sigmoid'))
	return model

# Generator with Discriminator
def generator_containing_discriminator(generator,discriminator):
	model = Sequential()
	model.add(generator)
	discriminator.trainable = False
	model.add(discriminator)
	return model

#### Training
discriminator = discriminator_model()
generator = generator_model()
discriminator_on_generator = generator_containing_discriminator(generator,discriminator)
# Optimizer
adam=Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
# Compile generator
generator.compile(loss='mean_squared_error',optimizer='nadam')
discriminator_on_generator.compile(loss='binary_crossentropy',optimizer=adam)
discriminator.trainable = True
discriminator.compile(loss='binary_crossentropy',optimizer='nadam')

for epoch in range(EPOCH):
	print("Epoch is", epoch,"of",EPOCH)
	print("Number of batches",int(F.shape[0]/BATCH_SIZE))
	start_time = time.time()


	for index in range(int(F.shape[0]/BATCH_SIZE)):
		image_batch = F[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
		BW_image_batch = G[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
		gAlone_loss = generator.train_on_batch(BW_image_batch,image_batch)
		#print("Generating images...")
		generated_images = generator.predict(BW_image_batch)

		# Creating inputs for train_on_batch
		M = numpy.concatenate((image_batch,generated_images))
		z = [1]*image_batch.shape[0]+[0]*generated_images.shape[0]
		# Shuffling M and z
		perm = numpy.random.permutation(len(z))
		M = M[perm]
		z = numpy.array(z)
		z = z[perm]
		#print("Training discriminator...")
		d_loss = discriminator.train_on_batch(M,z)

		for j in range(1):
			#print("Training generator...")
			g_loss = discriminator_on_generator.train_on_batch(BW_image_batch,[1]*BW_image_batch.shape[0])
			print("Generator loss %.4f"%gAlone_loss,"GAN loss %.4f "%g_loss, "Discriminator loss %.4f"%d_loss, "Total: %.4f"%(g_loss+d_loss+gAlone_loss),"For batch",index)
	# Test if discriminator is working
	print("DISCRIMINATOR_Imagem REAL=",discriminator.predict(image_batch)[index])
	print("DISCRIMINATOR_Imagem FAKE=",discriminator.predict(generator.predict(BW_image_batch))[index])
	print("GAN_Imagem FAKE=",discriminator_on_generator.predict(BW_image_batch)[index])

	print("Saving weights...")
	generator.save_weights(outDir+'/generator_weights',True)
	discriminator.save_weights(outDir+'/discriminator_weights',True)
	print("Saving images...")
	save3images(BW_image_batch,generated_images,image_batch,epoch)

	print("Elapsed time in epoch = ",str(timedelta(seconds=(time.time()-start_time))))
	print("----------------------------------")



# Print important parameters to logfile.log

print('Total samples = ', G.shape[0], ' Batch size =', BATCH_SIZE, ' Epochs = ', EPOCH)
print("Generator loss %.4f "%g_loss, "Discriminator loss %.4f"%d_loss, "Total: %.4f"%(g_loss+d_loss))
print("----------------------------------")
print("---DISCRIMINATOR---")
print(discriminator.summary())
print("----------------------------------")
print("---GENERATOR---")
print(generator.summary())
print("----------------------------------")
print("---GAN---")
print(discriminator_on_generator.summary())
print("----------------------------------")
# eof
