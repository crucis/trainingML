import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
import math
from scipy import misc as sc

def rgb2gray(rgb):
	return numpy.dot(rgb[...,:3], [0.299, 0.587, 0.114])
def converter(a,b):
	for i in range(0,b.shape[0]-1):
		a[i,0,:,:] = rgb2gray(b[i,:,:,:].transpose(1,2,0))

def rgb2yuv(rgb):
	yuv = numpy.zeros(rgb.shape)
	# Analog transformation
#	yuv[:,:,0] = 0.299*rgb[...,0]+0.587*rgb[...,1]+0.114*rgb[...,2]
#	yuv[:,:,1] = -0.14713*rgb[...,0]-0.28886*rgb[...,1]+0.436*rgb[...,2]
#	yuv[:,:,2] = 0.615*rgb[...,0]-0.51499*rgb[...,1]-0.10001*rgb[...,2]
	# Digital transformation using data range [0,255] BT 601
#	yuv[:,:,0] = 16 + (65.738/256)*rgb[...,0] + (129.057/256)*rgb[...,1]+(25.064/256)*rgb[...,2]
#	yuv[:,:,1] = 128 - (37.945/256)*rgb[...,0]-(74.494/256)*rgb[...,1]+(112.439/256)*rgb[...,2]
#	yuv[:,:,2] = 128 + (112.439/256)*rgb[...,0]-(94.154/256)*rgb[...1]-(18.285/256)*rgb[...,2]

	# Digital transformation using data range [0,255] BT 871
	yuv[:,:,0] = numpy.minimum(numpy.maximum(0,numpy.around(0 + 0.299*rgb[...,0]+0.587*rgb[...,1]+0.114*rgb[...,2])),255)
	yuv[:,:,1] = numpy.minimum(numpy.maximum(0,numpy.around(128 - 0.168736*rgb[...,0]-0.331264*rgb[...,1]+0.5*rgb[...,2])),255)
	yuv[:,:,2] = numpy.minimum(numpy.maximum(0,numpy.around(128 + 0.5*rgb[...,0]-0.418688*rgb[...,1]-0.081312*rgb[...,2])),255)

#	print("yuv.max=",numpy.amax(yuv),"yuv.min=",numpy.amin(yuv))
	return yuv.transpose(2,0,1)
#	return numpy.dot(rgb[:,:,0],[[0.299,0.587,0.114],[-0.14713,-0.28886,0.436],[0.615,-0.51499,-0.10001]])
def converterYUV(a,b):
	for i in range(0, b.shape[0]):
		h = b[i].transpose(1,2,0)
		a[i] = rgb2yuv(h)

def yuv2rgb(yuv):
	rgb = numpy.zeros(yuv.shape)
#	yuv*=255
#	print("yuv.shape=",yuv.shape)
#	print("yuv.max=",numpy.amax(yuv),"yuv.min=",numpy.amin(yuv))
	# Analog transformation
#	rgb[:,:,0] = yuv[...,0]+0*yuv[...,1]+1.13983*yuv[...,2]
#	rgb[:,:,1] = yuv[...,0]-0.39465*yuv[...,1]-0.58060*yuv[...,2]
#	rgb[:,:,2] = yuv[...,0]+2.03211*yuv[...,1]+0*yuv[...,2]

	# Digital transformation BT871
	rgb[:,:,0] = numpy.minimum(numpy.maximum(0,numpy.around(yuv[...,0]+1.402*(yuv[...,2]-128))),255)
	rgb[:,:,1] = numpy.minimum(numpy.maximum(0,numpy.around(yuv[...,0]-0.344136*(yuv[...,1]-128)-0.714136*(yuv[...,2]-128))),255)
	rgb[:,:,2] = numpy.minimum(numpy.maximum(0,numpy.around(yuv[...,0]+1.772*(yuv[...,1]-128))),255)
#	print("rgb.max=",numpy.amax(rgb),"rgb.min=",numpy.amin(rgb))
	return rgb.transpose(2,0,1)
def converterRGB(a,b):
#	print("b.shape=",b.shape)
	for i in range(0,b.shape[0]):
		a[i] = yuv2rgb(b[i].transpose(1,2,0))

def save3images(inp,out,original,folder):
#	out = numpy.concatenate((inp, out), axis=1)
#	converterRGB(out,out*255)
#	print("out.shape=",out.shape,"out.max",numpy.amax(out),"out.min=",numpy.amin(original))
#	original = numpy.concatenate((inp,original),axis=1)
#	converterRGB(original,original*255)
#	print("original.shape=",original.shape,"original.max",numpy.amax(original),"original.min=",numpy.amin(original))

	for i in range(int(numpy.around(original.shape[0]*0.02))):
		_,((ax1,ax2),(ax3,_)) = plt.subplots(2,2,sharey='row',sharex='col')

		n = numpy.around(uniform(0,original.shape[0]))

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


outDir='testeYUV'
im = sc.imread('results/Test8/samples/epoch49/sample_0.png')
save3images(im(:,:,0).transpose(2,0,1),yuv2rgb(rgb2yuv(im.transpose(2,0,1))),im.transpose(2,0,1))
