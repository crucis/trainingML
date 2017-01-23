from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, Deconvolution2D
from keras import backend as K
from random import uniform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
import math
import sys
import time
from datetime import timedelta
from scipy import misc



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
#BATCH_SIZE = 256
#EPOCH = 50
nImages = 40

#### CIFAR10 classifications
cifar10_Classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck','all'];

# Model Options
folder = "Test28"
outDire = 'results/'+folder

#d_predict_fake = 0

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
	for i in range(0,b.shape[0]-1):
		a[i,0,:,:] = rgb2gray(b[i,:,:,:].transpose(1,2,0))

def rgb2yuv(rgb):
	rgb = rgb.astype('float32')
	yuv = numpy.zeros(rgb.shape)
	yuv = yuv.astype('float32')

	# Digital transformation using data range [0,255] BT 871
	yuv[:,:,0] = numpy.minimum(numpy.maximum(0,numpy.around(0 + 0.299*rgb[...,0]+0.587*rgb[...,1]+0.114*rgb[...,2])),255)
	yuv[:,:,1] = numpy.minimum(numpy.maximum(0,numpy.around(128 - 0.168736*rgb[...,0]-0.331264*rgb[...,1]+0.5*rgb[...,2])),255)
	yuv[:,:,2] = numpy.minimum(numpy.maximum(0,numpy.around(128 + 0.5*rgb[...,0]-0.418688*rgb[...,1]-0.081312*rgb[...,2])),255)

	return yuv.transpose(2,0,1)
def converterYUV(a,b):
	for i in range(0, b.shape[0]):
		h = b[i].transpose(1,2,0)
		a[i] = rgb2yuv(h)

def yuv2rgb(yuv):
	yuv = yuv.astype('float32')
	rgb = numpy.zeros(yuv.shape)
	rgb = rgb.astype('float32')

	# Digital transformation BT871
	rgb[:,:,0] = numpy.minimum(numpy.maximum(0,numpy.around(yuv[...,0]+1.402*(yuv[...,2]-128))),255)
	rgb[:,:,1] = numpy.minimum(numpy.maximum(0,numpy.around(yuv[...,0]-0.344136*(yuv[...,1]-128)-0.714136*(yuv[...,2]-128))),255)
	rgb[:,:,2] = numpy.minimum(numpy.maximum(0,numpy.around(yuv[...,0]+1.772*(yuv[...,1]-128))),255)

	return rgb.transpose(2,0,1)
def converterRGB(a,b):
#	print("b.shape=",b.shape)
	for i in range(0,b.shape[0]):
		a[i] = yuv2rgb(b[i].transpose(1,2,0))


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
		mkdir_p(output_dir) # Checks if directory exists, and creates it if necessary
		figurestr = output_dir+filename+"_"+nameVec+".png"
		plt.savefig(figurestr)
	if SHOW_GRAPHS == 1:
		plt.show()
	plt.clf()
	plt.close('all')

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
	out = numpy.concatenate((inp, out), axis=1)
	converterRGB(out,out*255+128)
	original = numpy.concatenate((inp,original),axis=1)
	converterRGB(original,original*255+128)

	for i in range(original.shape[0]):
		_,((ax1,ax2),(ax3,_)) = plt.subplots(2,2,sharey='row',sharex='col')

	#	n = math.floor(uniform(0,original.shape[0]))

		ax1.imshow(inp[i,0,:,:],cmap='gray')
		ax1.set_title('Input_%s'%i)

		ax2.imshow(numpy.uint8(out[i].transpose(1,2,0)))
		ax2.set_title('Output_%s'%i)

		ax3.imshow(numpy.uint8(original[i].transpose(1,2,0)))
		ax3.set_title('Original_%s'%i)

		titlestr = 'Sample'
		plt.title(titlestr)
		plt.grid(b=False)

		mkdir_p(outDir+'/samples/'+str(folder))

		plt.savefig(outDir+'/samples/'+str(folder)+'/sample_%s.png'%i)
		plt.clf()
		plt.close('all')

def plotHistogram(grayImage,originalImage,fakeImage, nameClass,directory,folder=folder):
	# Faz 3 histogramas, um para azul outro verde e outro vermelho
	fakeImage = numpy.concatenate((grayImage, fakeImage), axis=1)
	converterRGB(fakeImage,fakeImage*255+128)
	originalImage = numpy.concatenate((grayImage,originalImage),axis=1)
	converterRGB(originalImage,originalImage*255+128)

	name = folder+'using'+nameClass
	originalImage = originalImage.astype('uint8')
	fakeImage = fakeImage.astype('uint8')
	numBins = 255

	for i in range(originalImage.shape[0]):
		if i == 0:
			originalRed2D = numpy.array(originalImage[i,0,:,:])
			originalGreen2D = numpy.array(originalImage[i,1,:,:])
			originalBlue2D = numpy.array(originalImage[i,2,:,:])
		else:
			originalRed2D = numpy.vstack([originalRed2D,originalImage[i,0,:,:]])
			originalGreen2D = numpy.vstack([originalGreen2D,originalImage[i,1,:,:]])
			originalBlue2D = numpy.vstack([originalBlue2D,originalImage[i,2,:,:]])

	fO, (ax1,ax2,ax3) = plt.subplots(3, sharex=True,sharey=True)
	ax1.hist(originalRed2D.ravel(), color='red', bins=numBins,alpha=0.8, align='mid')
	ax1.set_title('Original images Histogram '+name)
	ax2.hist(originalGreen2D.ravel(),color='green', bins=numBins,alpha=0.8, align='mid')
	ax3.hist(originalBlue2D.ravel(),color='blue', bins=numBins,alpha=0.8,  align='mid')
	plt.xlim(0,256)
	plt.savefig(directory+'/histOriginal.png')
	plt.clf()


	for j in range(fakeImage.shape[0]):
		if j == 0:
			generatedRed2D = numpy.array(fakeImage[j,0,:,:])
			generatedGreen2D = numpy.array(fakeImage[j,1,:,:])
			generatedBlue2D = numpy.array(fakeImage[j,2,:,:])
		else:
			generatedRed2D = numpy.vstack([generatedRed2D,fakeImage[j,0,:,:]])
			generatedGreen2D = numpy.vstack([generatedGreen2D,fakeImage[j,1,:,:]])
			generatedBlue2D = numpy.vstack([generatedBlue2D,fakeImage[j,2,:,:]])


	fO, (ax1,ax2,ax3) = plt.subplots(3, sharex=True,sharey=True)
	ax1.hist(generatedRed2D.ravel(),color='red', bins=numBins,alpha=0.8, align='mid')
	ax1.set_title('Generated images Histogram '+name)
	ax2.hist(generatedGreen2D.ravel(),color='green',bins=numBins, alpha=0.8, align='mid')
	ax3.hist(generatedBlue2D.ravel(),color='blue',bins=numBins, alpha=0.8, align='mid')
	plt.xlim(0,256)
	plt.savefig(directory+'/histGenerated.png')
	plt.clf()
	plt.close('all')

########################
#PROGRAM
########################
#### Models
# GENERATOR
def generator_model():
	model = Sequential()
	# Layers
	model.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal', input_shape=(1, 32, 32)))
	model.add(LeakyReLU(0.2))

	model.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal'))
	model.add(BatchNormalization(mode=2,axis=1))
	model.add(LeakyReLU(0.2))

	model.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal'))
	model.add(BatchNormalization(mode=2,axis=1))
	model.add(LeakyReLU(0.2))

	model.add(Convolution2D(256, 3, 3, border_mode='same', init='he_normal'))
	model.add(BatchNormalization(mode=2,axis=1))
	model.add(LeakyReLU(0.2))

	model.add(Convolution2D(256,3,3,border_mode='same',init='he_normal'))
	model.add(BatchNormalization(mode=2,axis=1))
	model.add(LeakyReLU(0.2))

	model.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal'))
	model.add(BatchNormalization(mode=2,axis=1))
	model.add(LeakyReLU(0.2))
	#model.add(BatchNormalization())

	model.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal'))
	model.add(BatchNormalization(mode=2,axis=1))
	model.add(LeakyReLU(0.2))

	model.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal'))
	model.add(BatchNormalization(mode=2,axis=1))
	model.add(LeakyReLU(0.2))

	model.add(Convolution2D(2, 3, 3, border_mode='same', init='he_normal'))
	model.add(Lambda(lambda x: K.clip(x, 0.0, 1.0)))
	return model

# DISCRIMINATOR
def discriminator_model():
	model = Sequential()
	model.add(Convolution2D(32,3,3,border_mode='same',init='he_normal',input_shape=(2,32,32),subsample=(2,2))) #16x16
	model.add(LeakyReLU(alpha=.2))

	model.add(Convolution2D(64,3,3,border_mode='same',init='he_normal',subsample=(2,2))) #8x8
	model.add(LeakyReLU(alpha=.2))

	model.add(Convolution2D(128,3,3,border_mode='same',init='he_normal',subsample=(2,2))) #4x4
	model.add(LeakyReLU(alpha=.2))
	model.add(Dropout(0.2))

	model.add(Convolution2D(256,3,3,border_mode='same',init='he_normal',subsample=(2,2))) #2x2
	model.add(LeakyReLU(alpha=.2))
	model.add(Dropout(0.2))

	model.add(Flatten())
	model.add(Dense(512,init='he_normal'))
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

#for i in range(7,len(cifar10_Classes)):
#chosen_Class = cifar10_Classes[i]
chosen_Class = 'all'
outDir = outDire+'/'+str(chosen_Class)
# Create folder for tests
mkdir_p(outDir)
#	sys.stdout = Logger()

# Logger
#### load cifar10 dataset
(Y_rgb,labels),(Y_rgb_test, labels_test) = cifar10.load_data()
#Y_rgb = numpy.zeros((8,32,32,3))
#Y_rgb[0,...] = misc.imread('images/0.jpg')
#Y_rgb[1,...] = misc.imread('images/1.jpg')
#Y_rgb[2,...] = misc.imread('images/2.jpg')
#Y_rgb[3,...] = misc.imread('images/3.jpg')
#Y_rgb[4,...] = misc.imread('images/4.jpg')
#Y_rgb[5,...] = misc.imread('images/5.jpg')
#Y_rgb[6,...] = misc.imread('images/6.jpg')
#Y_rgb[7,...] = misc.imread('images/7.jpg')

#Y_rgb = Y_rgb.transpose(0,3,1,2)

#Y_rgb_test=Y_rgb
# Choosing only one classification
#if str(chosen_Class) != 'all':
#	Y_rgb = Y_rgb[(labels == cifar10_Classes.index(chosen_Class))[:,0]]
#	Y_rgb_test = Y_rgb_test[(labels_test == cifar10_Classes.index(chosen_Class))[:,0]]

Y_yuv = numpy.zeros((Y_rgb.shape[0],Y_rgb.shape[1],Y_rgb.shape[2],Y_rgb.shape[3]))
Y_yuv_test = numpy.zeros((Y_rgb_test.shape[0],Y_rgb.shape[1],Y_rgb_test.shape[2],Y_rgb_test.shape[3]))

# Convert RGB to YUV to create our input
converterYUV(Y_yuv,Y_rgb)
converterYUV(Y_yuv_test,Y_rgb_test)

# Separate grayscale channel from U and V channels
Y_gray = numpy.zeros((Y_yuv.shape[0],1,Y_yuv.shape[2],Y_yuv.shape[3]))
Y_gray[:,0,:,:] = Y_yuv[:,0,:,:]
Y_uv = numpy.zeros((Y_yuv.shape[0],2,Y_yuv.shape[2],Y_yuv.shape[3]))
Y_uv = Y_yuv[:,1:,:,:]
Y_gray_test = numpy.zeros((Y_yuv_test.shape[0],1,Y_yuv_test.shape[2],Y_yuv_test.shape[3]))
Y_gray_test[:,0,:,:] = Y_yuv_test[:,0,:,:]
Y_uv_test = numpy.zeros((Y_yuv_test.shape[0],2,Y_yuv_test.shape[2],Y_yuv_test.shape[3]))
Y_uv_test = Y_yuv_test[:,1:,:,:]

#### Data Preprocessing
# convert inputs and outputs to float32
Y_gray = Y_gray.astype('float32')
Y_uv = Y_uv.astype('float32')
Y_gray_test = Y_gray_test.astype('float32')
Y_uv_test = Y_uv_test.astype('float32')
# normalize inputs and outputs from 0-255 to 0.0-1.0
	# normalize inputs and outputs from 0-255 to -1.0-1.0
Y_gray = Y_gray/255 - 128
Y_uv = Y_uv/255 - 128
Y_gray_test = Y_gray_test/255 - 128
Y_uv_test = Y_uv_test/255 - 128


# limits the number of images to nImages
#perm = numpy.random.permutation(Y_gray.shape[0])
perm = numpy.random.permutation(Y_gray_test.shape[0])

Y_gray = Y_gray[perm]
Y_uv = Y_uv[perm]
Y_gray_test = Y_gray_test[perm]
Y_uv_test = Y_uv_test[perm]



G = Y_gray[:nImages]
F = Y_uv[:nImages]
G_test = Y_gray_test[:nImages]
F_test = Y_uv_test[:nImages]


print("----------------------------------")
print('Loading with dataset based on class - ',chosen_Class,'with',F.shape[0],'samples')
print("----------------------------------")

#generator = generator_model()
#generator.load_weights(outDir+"/generator_weights")
generator = load_model(outDir+'/model.h5')

print("Saving sample images...")
g_predict_fake = generator.predict(G_test)
save3images(G_test,g_predict_fake,F_test,"Saved_images/")

#if chosen_Class == 'all':
#	print('Saving histograms')
#	stored_g_predict = generator.predict(Y_gray_test)
#	plotHistogram(grayImage=Y_gray_test,originalImage=Y_uv_test,fakeImage=stored_g_predict,nameClass = chosen_Class, directory=outDir)
#	print("----------------------------------")


# eof
