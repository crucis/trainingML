from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
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
#import subprocess




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
EPOCH = 1
nImages = pow(2,10)

# Model Options
folder = "Test15"
outDir = 'results/'+folder

#### CIFAR10 classifications
cifar10_Classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'];
chosen_Class = 'ship'

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
def plotHistogram(originalImage,fakeImage, nameClass = chosen_Class,directory = outDir,folder=folder):
	# Faz 3 histogramas, um para azul outro verde e outro vermelho
	name = folder+'using'+nameClass
	originalImage *= 255
	fakeImage *= 255
	originalImage = originalImage.astype('uint8')
	fakeImage = fakeImage.astype('uint8')
	numBins = 256

	fO, (ax1,ax2,ax3) = plt.subplots(3, sharex=True,sharey=True)
	ax1.hist(originalImage[:,0,:,:],numBins,color='red',alpha=0.8)
	ax1.set_title('Original images Histogram '+name)
	ax2.hist(originalImage[:,1,:,:],numBins,color='green',alpha=0.8)
	ax3.hist(originalImage[:,2,:,:],numBins,color='blue',alpha=0.8)
	plt.savefig(directory+'/OriginalHist.png')
	plt.clf()

	f1, (ax1,ax2,ax3) = plt.subplots(3, sharex=True,sharey=True)
	ax1.hist(originalImage[:,0,:,:],numBins,color='red',alpha=0.8)
	ax1.set_title('Generated images Histogram '+name)
	ax2.hist(originalImage[:,1,:,:],numBins,color='green',alpha=0.8)
	ax3.hist(originalImage[:,2,:,:],numBins,color='blue',alpha=0.8)
	plt.savefig(directory+'/GeneratedHist.png')
	plt.clf()

########################
#PROGRAM
########################
# Create folder for tests
mkdir_p(outDir)
# Logger
sys.stdout = Logger()
#### load cifar10 dataset
(Y,labels),(Y_test, labels_test) = cifar10.load_data()


# Choosing only one classification
Y = Y[(labels == cifar10_Classes.index(chosen_Class))[:,0]]
#Y_test = Y_test[(labels_test == cifar10_Classes.index(chosen_Class))[:,0]]

X = numpy.zeros((Y.shape[0],1,Y.shape[2],Y.shape[3]))
#X_test = numpy.zeros((Y_test.shape[0],1,Y.shape[2],Y.shape[3]))

# Convert RGB to grayscale to create our input
converter(X,Y)
#converter(X_test,Y_test)

#### Data Preprocessing
# convert inputs and outputs to float32
X = X.astype('float32')
Y = Y.astype('float32')
#X_test = X_test.astype('float32')
#Y_test = Y_test.astype('float32')
# normalize inputs and outputs from 0-255 to 0.0-1.0
X /= 255
Y /= 255
#X_test /= 255
#Y_test /= 255


# limits the number of images to nImages
G = X[:nImages]
F = Y[:nImages]
#G_test = X_test[nImages:nImages+math.ceil(nImages/5)]
#F_test = Y_test[nImages:nImages+math.ceil(nImages/5)]


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
	model.add(Convolution2D(8,3,3,border_mode='same',init='he_normal',input_shape=(3,32,32),activation='linear',W_regularizer = l2(0.001)))
	model.add(LeakyReLU(alpha=.2))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Convolution2D(16,3,3,border_mode='same',init='he_normal',activation='linear',W_regularizer = l2(0.001)))
	model.add(LeakyReLU(alpha=.2))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Convolution2D(32,3,3,border_mode='same',init='he_normal',activation='linear',W_regularizer = l2(0.001)))
	model.add(LeakyReLU(alpha=.2))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Convolution2D(64,3,3,border_mode='same',init='he_normal',activation='linear',W_regularizer = l2(0.001)))
	model.add(LeakyReLU(alpha=.2))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Flatten())
	model.add(Dense(512,init='he_normal',activation='linear',W_regularizer = l2(0.001)))
	model.add(LeakyReLU(alpha=.2))
	model.add(Dropout(0.2))
	model.add(Dense(256,init='he_normal',activation='linear',W_regularizer = l2(0.001)))
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
discriminator.load_weights("results/Test12/discriminator_weights")
generator = generator_model()
# LOADING GENERATOR FROM TEST8
#generator.load_weights("results/Test8/generator_weights")
discriminator_on_generator = generator_containing_discriminator(generator,discriminator)
# Optimizer
adam=Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
# Compile generator
generator.compile(loss='mean_squared_error',optimizer='nadam')
discriminator_on_generator.compile(loss='binary_crossentropy',optimizer=adam)
discriminator.trainable = True
discriminator.compile(loss='binary_crossentropy',optimizer='nadam', metrics=['accuracy'])

# Initialize d_loss
d_predict_real = 0
d_loss = 1
d_acc = 0

for epoch in range(EPOCH):
	print("Epoch is", epoch+1,"of",EPOCH)
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
        # Does not train discriminator if it is close to overfitting
		if d_acc > 0.55:
			discriminator.trainable = False
		else:
			discriminator.trainable = True
			[d_loss, d_acc] = discriminator.train_on_batch(M,z)



		for j in range(1):
			#print("Training generator...")
			g_loss = discriminator_on_generator.train_on_batch(BW_image_batch,[1]*BW_image_batch.shape[0])
			print("GAN loss %.4f "%g_loss, "Discriminator loss %.4f"%d_loss,"Discriminator accuracy %.4f"%d_acc,"Generator loss %.4f"%gAlone_loss, "Total: %.4f"%(g_loss+d_loss),"For batch",index)
            #print("Generator loss %.4f"%gAlone_loss,"GAN loss %.4f "%g_loss, "Discriminator loss %.4f"%d_loss, "Total: %.4f"%(g_loss+d_loss+gAlone_loss),"For batch",index)
	# Test if discriminator is working
	d_predict_real = discriminator.predict(image_batch)
	print("DISCRIMINATOR_Imagem REAL=",d_predict_real[index])
	g_predict_fake = generator.predict(BW_image_batch)
	print("DISCRIMINATOR_Imagem FAKE=",discriminator.predict(g_predict_fake)[index])
	print("GAN_Imagem FAKE=",discriminator_on_generator.predict(BW_image_batch)[index])

	print("Saving weights...")
	generator.save_weights(outDir+'/generator_weights',True)
	discriminator.save_weights(outDir+'/discriminator_weights',True)
	print("Saving sample images...")
	save3images(BW_image_batch,generated_images,image_batch,epoch)
	print("Storing to histogram values")
	if index == 0:
		stored_g_predict = numpy.array(g_predict_fake)
	else:
		stored_g_predict = numpy.vstack(g_predict_fake)
	print("Elapsed time in epoch = ",str(timedelta(seconds=(time.time()-start_time))))
	print("----------------------------------")



# Print important parameters to logfile.log and save histogram
print('End of training')
print('Saving histograms')
plotHistogram(originalImage=F,fakeImage=stored_g_predict)
print("----------------------------------")

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
#subprocess.call(['speech-dispatcher'])        #start speech dispatcher
#subprocess.call(['spd-say', '"your process has finished"'])
# eof
