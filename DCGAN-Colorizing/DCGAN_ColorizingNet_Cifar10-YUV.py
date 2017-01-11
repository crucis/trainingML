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
import matplotlib
matplotlib.use('Agg')
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
EPOCH = 50
nImages = pow(2,15)

#### CIFAR10 classifications
cifar10_Classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'];
#chosen_Class = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'] # Each chosen class from cifar10_Classes is a loop, if 'all' chosen, than it will run the entire cifar10 >>NOT IMPLEMENTED

# Model Options
folder = "Test21"
outDire = 'results/'+folder

d_predict_fake = 0

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

def rgb2yuv(rgb):
	yuv = numpy.zeros(rgb.shape)
	yuv[:,:,0] = 0.299*rgb[...,0]+0.587*rgb[...,1]+0.114*rgb[...,2]
	yuv[:,:,1] = -0.14713*rgb[...,0]-0.28886*rgb[...,1]+0.436*rgb[...,2]
	yuv[:,:,2] = 0.615*rgb[...,0]-0.51499*rgb[...,1]-0.10001*rgb[...,2]
	return yuv.transpose(2,0,1)
#	return numpy.dot(rgb[:,:,0],[[0.299,0.587,0.114],[-0.14713,-0.28886,0.436],[0.615,-0.51499,-0.10001]])
def converterYUV(a,b):
	for i in range(0, b.shape[0]):
		h = b[i].transpose(1,2,0)
		a[i] = rgb2yuv(h)

def yuv2rgb(yuv):
	rgb = numpy.zeros(yuv.shape)
	rgb[:,:,0] = yuv[...,0]+1.13983*yuv[...,2]
	rgb[:,:,1] = yuv[...,0]-0.39465*yuv[...,1]-0.58060*yuv[...,2]
	rgb[:,:,2] = yuv[...,0]+2.03211*yuv[...,1]
	return yuv.transpose(2,0,1)
def converterRGB(a,b):
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
		mkdir_p(output_dir) # Verifies if directory exists, and creates it if necessary
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
	converterRGB(out,out)
	original = numpy.concatenate((inp,original),axis=1)
	converterRGB(original,original)
	for i in range(int(math.floor(original.shape[0]*0.02))):
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
		plt.close('all')
def plotHistogram(originalImage,fakeImage, nameClass,directory,folder=folder):
	# Faz 3 histogramas, um para azul outro verde e outro vermelho
	name = folder+'using'+nameClass
	originalImage *= 255
	fakeImage *= 255
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
	#model.add(BatchNormalization())

	model.add(Convolution2D(256, 3, 3, border_mode='same', init='he_normal'))
        model.add(BatchNormalization(mode=2,axis=1))
        model.add(LeakyReLU(0.2))
	#model.add(BatchNormalization())

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
	#model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.2))

	model.add(Convolution2D(64,3,3,border_mode='same',init='he_normal',subsample=(2,2))) #8x8
	model.add(LeakyReLU(alpha=.2))
	#model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.2))

	model.add(Convolution2D(128,3,3,border_mode='same',init='he_normal',subsample=(2,2))) #4x4
	model.add(LeakyReLU(alpha=.2))
	#model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.2))

	model.add(Convolution2D(256,3,3,border_mode='same',init='he_normal',subsample=(2,2))) #2x2
	model.add(LeakyReLU(alpha=.2))
	#model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.2))

	model.add(Flatten())
	#model.add(Dense(128,init='he_normal'))
	model.add(Dense(1024,init='he_normal'))
	model.add(LeakyReLU(alpha=.2))
	model.add(Dropout(0.2))

	#model.add(Dense(256,init='he_normal',activation='linear',W_regularizer = l2(0.001)))
	#model.add(LeakyReLU(alpha=.2))
	#model.add(Dropout(0.2))
	model.add(Dense(1,init='he_normal',activation='sigmoid'))
	return model

# Generator with Discriminator
def generator_containing_discriminator(generator,discriminator):
	model = Sequential()
	model.add(generator)
	discriminator.trainable = False
	model.add(discriminator)
	return model

for i in range(7,len(cifar10_Classes)):
	chosen_Class = cifar10_Classes[i]
	outDir = outDire+'/'+str(chosen_Class)
	# Create folder for tests
	mkdir_p(outDir)
	sys.stdout = Logger()

	# Logger
	#### load cifar10 dataset
	(Y_rgb,labels),(Y_rgb_test, labels_test) = cifar10.load_data()


	# Choosing only one classification
	Y_rgb = Y_rgb[(labels == cifar10_Classes.index(chosen_Class))[:,0]]
	Y_rgb_test = Y_rgb_test[(labels_test == cifar10_Classes.index(chosen_Class))[:,0]]

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
	Y_gray /= 255
	Y_uv /= 255
	Y_gray_test /= 255
	Y_uv_test /= 255


	# limits the number of images to nImages
	G = Y_gray[:nImages]
	F = Y_uv[:nImages]
	G_test = Y_gray_test[:nImages]
	F_test = Y_uv_test[:nImages]


	print("----------------------------------")
	print('Training with dataset based on class - ',chosen_Class,'with',F.shape[0],'samples')
	print("----------------------------------")




	#### Training
	discriminator = discriminator_model()
#	discriminator.load_weights("results/PreTrainedWeights1/"+chosen_Class+"/discriminator_weights")
	generator = generator_model()
	# LOADING GENERATOR FROM TEST8
	#generator.load_weights("results/PreTrainedWeights1/"+chosen_Class+"/generator_weights")
	#generator.load_weights("results/Test17/"+chosen_Class+"/generator_weights")

	discriminator_on_generator = generator_containing_discriminator(generator,discriminator)
	# Optimizer
	adam=Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
	# Compile generator
	generator.compile(loss='mean_squared_error',optimizer='adam')
	discriminator_on_generator.compile(loss='binary_crossentropy',optimizer=adam, metrics=['accuracy'])
        discriminator_on_generator.summary()
	discriminator.trainable = True
	discriminator.compile(loss='binary_crossentropy',optimizer=adam, metrics=['accuracy'])
        discriminator.summary()

	# Initialize d_loss
	d_predict_real = 0
	d_loss = 1
	d_acc = 0
	d_predict_fake = 0
	d_predict_real = 1
	 #counter for how many times discriminator not trained

	for epoch in range(EPOCH):
		print("Epoch", epoch+1,"of",EPOCH)
		print("Number of batches",int(F.shape[0]/BATCH_SIZE))
		start_time = time.time()
		l=0 # Counts how many times discriminator was NOT trained
		m = 0; #Counts batchs
		for index in range(int(F.shape[0]/BATCH_SIZE)):
			image_batch = F[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
			BW_image_batch = G[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
			#image_batch_test = F_test[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
			#BW_image_batch_test = G_test[index*BATCH_SIZE:(index+1)*BATCH_SIZE]

			#gAlone_loss = generator.train_on_batch(BW_image_batch,image_batch)
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
			if (numpy.mean(d_predict_fake) > 0.50) or (numpy.mean(d_predict_real) < 0.5):
				discriminator.trainable = True
			elif (d_acc > 0.95) :
				discriminator.trainable = False
				l+=1
			else:
				discriminator.trainable = True
			[d_loss, d_acc] = discriminator.train_on_batch(M,z)


			for j in range(1):
				#print("Training generator...")
				[g_loss,g_acc] = discriminator_on_generator.train_on_batch(BW_image_batch,[1]*BW_image_batch.shape[0])
				print("GAN loss %.4f "%g_loss, "GAN acc %.4f"%g_acc, "Discriminator loss %.4f"%d_loss,"Discriminator accuracy %.4f"%d_acc, "Total loss: %.4f"%(g_loss+d_loss),"for batch",index)
				#print("GAN loss %.4f "%g_loss, "Discriminator loss %.4f"%d_loss,"Discriminator accuracy %.4f"%d_acc, "Total loss: %.4f"%(g_loss+d_loss),"for batch",index)
	            #print("Generator loss %.4f"%gAlone_loss,"GAN loss %.4f "%g_loss, "Discriminator loss %.4f"%d_loss, "Total: %.4f"%(g_loss+d_loss+gAlone_loss),"For batch",index)
		# Test if discriminator is working
		d_predict_real = discriminator.predict(F_test)
		#print(len(d_predict_real))
		print("DISCRIMINATOR_Imagem REAL=",numpy.mean(d_predict_real))
		g_predict_fake = generator.predict(G_test)
		d_predict_fake = discriminator.predict(g_predict_fake)
		print("DISCRIMINATOR_Imagem FAKE=",numpy.mean(d_predict_fake))
	#	print("GAN_Imagem FAKE=",numpy.mean(discriminator_on_generator.predict(G_test)))
		print("Discriminator trained",index-l+1,"times of",index+1,"batchs")

		print("Saving weights...")
		generator.save_weights(outDir+'/generator_weights',True)
		discriminator.save_weights(outDir+'/discriminator_weights',True)
		print("Saving sample images...")
		save3images(BW_image_batch,generated_images,image_batch,epoch+1)
		print("Storing to histogram values")


		print("Elapsed time in epoch = ",str(timedelta(seconds=(time.time()-start_time))))
		print("----------------------------------")
		m+=1


	# Print important parameters to logfile.log and save histogram
	#print('fakeImage=',stored_g_predict.shape)
	#print('originalImage=',F.shape)
	print('End of training')
	print('Saving histograms')
	stored_g_predict = generator.predict(G)
	plotHistogram(originalImage=F,fakeImage=stored_g_predict,nameClass = chosen_Class, directory=outDir)
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
	#print("----------------------------------")

	# eof
