import tensorflow.contrib.slim as slim
import scipy.misc
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import shutil
import utils
import cv2
import numpy as np
import os
import data
import random

"""
An implementation of the neural network used for
super-resolution of images as described in:

`Enhanced Deep Residual Networks for Single Image Super-Resolution`
(https://arxiv.org/pdf/1707.02921.pdf)

(single scale baseline-style model)
"""
class EDSR(object):

	def __init__(self,img_size=32,num_layers=32,feature_size=256,scale=2,output_channels=3):
		print("Building EDSR...")
		self.img_size = img_size
		self.scale = scale
		self.output_channels = output_channels

		#Placeholder for image inputs
		self.input = x = tf.placeholder(tf.float32,[None,img_size,img_size,output_channels])
		#Placeholder for upscaled image ground-truth
		self.target = y = tf.placeholder(tf.float32,[None,img_size*scale,img_size*scale,output_channels])
	
		"""
		Preprocessing as mentioned in the paper, by subtracting the mean
		However, the subtract the mean of the entire dataset they use. As of
		now, I am subtracting the mean of each batch
		"""
		mean_x = tf.reduce_mean(self.input)#127
		image_input =x- mean_x
		mean_y = tf.reduce_mean(self.target)#127
		image_target =y- mean_y

		#One convolution before res blocks and to convert to required feature depth
		x = slim.conv2d(image_input,feature_size,[3,3])
	
		#Store the output of the first convolution to add later
		conv_1 = x	

		"""
		This creates `num_layers` number of resBlocks
		a resBlock is defined in the paper as
		(excuse the ugly ASCII graph)
		x
		|\
		| \
		|  conv2d
		|  relu
		|  conv2d
		| /
		|/
		+ (addition here)
		|
		result
		"""

		"""
		Doing scaling here as mentioned in the paper:

		`we found that increasing the number of feature
		maps above a certain level would make the training procedure
		numerically unstable. A similar phenomenon was
		reported by Szegedy et al. We resolve this issue by
		adopting the residual scaling with factor 0.1. In each
		residual block, constant scaling layers are placed after the
		last convolution layers. These modules stabilize the training
		procedure greatly when using a large number of filters.
		In the test phase, this layer can be integrated into the previous
		convolution layer for the computational efficiency.'

		"""
		scaling_factor = 0.1
		
		#Add the residual blocks to the model
		for i in range(num_layers):
			x = utils.resBlock(x,feature_size,scale=scaling_factor)#0.32-scaling_factor*0.1*i

		#One more convolution, and then we add the output of our first conv layer
		x = slim.conv2d(x,feature_size,[3,3])
		x += conv_1
		
		#Upsample output of the convolution		
		x = utils.upsample(x,scale,feature_size,None)

		#One final convolution on the upsampling output
		output = x#slim.conv2d(x,output_channels,[3,3])
		self.out = tf.clip_by_value(output+mean_x,0.0,255.0)
		# start------define my kernel sharp loss
		kernel_sharpen_4 = tf.constant([0, -1, 0, -1, 5, -1, 0, -1, 0], dtype=tf.float32)
		h1,h2,h3,h4 = y.shape
		kernel_sharpen_4 = tf.reshape(kernel_sharpen_4,(3,3,1,1))
		kernel_sharpen_4 = tf.tile(kernel_sharpen_4,(1,1,h4,h4))
		imglaplace = tf.nn.conv2d(y, kernel_sharpen_4, strides=[1,1,1,1], padding='SAME')
		outputlap =  tf.nn.conv2d(self.out, kernel_sharpen_4, strides=[1,1,1,1], padding='SAME')
		# end-------define my kernel sharp loss
		self.loss = loss = 0.8*tf.reduce_mean(tf.losses.absolute_difference(image_target,output)) + 0.2*tf.reduce_mean(tf.losses.absolute_difference(imglaplace,outputlap))
    #SRCNN : loss = tf.reduce_mean(tf.square(self.labels - self.pred))
    #LapSRN: loss = tf.reduce_mean(tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(tensor1,tensor2))+epsilon), [1, 2, 3]))
	
		#Calculating Peak Signal-to-noise-ratio
		#Using equations from here: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
		mse = tf.reduce_mean(tf.squared_difference(image_target,output))	
		PSNR = tf.constant(255**2,dtype=tf.float32)/mse
		PSNR = tf.constant(10,dtype=tf.float32)*utils.log10(PSNR)
	
		#Scalar to keep track for loss
		tf.summary.scalar("loss",self.loss)
		tf.summary.scalar("PSNR",PSNR)
		#Image summaries for input, target, and output
		tf.summary.image("input_image",tf.cast(self.input,tf.uint8))
		tf.summary.image("target_image",tf.cast(self.target,tf.uint8))
		tf.summary.image("output_image",tf.cast(self.out,tf.uint8))
		
		#Tensorflow graph setup... session, saver, etc.
		self.sess = tf.Session()
		self.saver = tf.train.Saver()
		print("Done building!")
	
	"""
	Save the current state of the network to file
	"""
	def save(self,savedir='saved_models'):
		print("Saving...")
		self.saver.save(self.sess,savedir+"/model")
		print("Saved!")
		
	"""
	Resume network from previously saved weights
	"""
	def resume(self,savedir='saved_models'):
		print("Restoring...")
		self.saver.restore(self.sess,tf.train.latest_checkpoint(savedir))
		print("Restored!")	

	"""
	Compute the output of this network given a specific input

	x: either one of these things:
		1. A numpy array of shape [image_width,image_height,3]
		2. A numpy array of shape [n,input_size,input_size,3]

	return: 	For the first case, we go over the entire image and run super-resolution over windows of the image
			that are of size [input_size,input_size,3]. We then stitch the output of these back together into the
			new super-resolution image and return that

	return  	For the second case, we return a numpy array of shape [n,input_size*scale,input_size*scale,3]
	"""
	def predict(self,x):
		print("Predicting...")
		if (len(x.shape) == 3) and not(x.shape[0] == self.img_size and x.shape[1] == self.img_size):
			num_across = x.shape[0]//self.img_size
			num_down = x.shape[1]//self.img_size
			tmp_image = np.zeros([x.shape[0]*self.scale,x.shape[1]*self.scale,3])
			for i in range(num_across):
				for j in range(num_down):
					tmp = self.sess.run(self.out,feed_dict={self.input:[x[i*self.img_size:(i+1)*self.img_size,j*self.img_size:(j+1)*self.img_size]]})[0]
					tmp_image[i*tmp.shape[0]:(i+1)*tmp.shape[0],j*tmp.shape[1]:(j+1)*tmp.shape[1]] = tmp
			#this added section fixes bottom right corner when testing
			if (x.shape[0]%self.img_size != 0 and  x.shape[1]%self.img_size != 0):
				tmp = self.sess.run(self.out,feed_dict={self.input:[x[-1*self.img_size:,-1*self.img_size:]]})[0]
				tmp_image[-1*tmp.shape[0]:,-1*tmp.shape[1]:] = tmp
					
			if x.shape[0]%self.img_size != 0:
				for j in range(num_down):
					tmp = self.sess.run(self.out,feed_dict={self.input:[x[-1*self.img_size:,j*self.img_size:(j+1)*self.img_size]]})[0]
					tmp_image[-1*tmp.shape[0]:,j*tmp.shape[1]:(j+1)*tmp.shape[1]] = tmp
			if x.shape[1]%self.img_size != 0:
				for j in range(num_across):
                                        tmp = self.sess.run(self.out,feed_dict={self.input:[x[j*self.img_size:(j+1)*self.img_size,-1*self.img_size:]]})[0]
                                        tmp_image[j*tmp.shape[0]:(j+1)*tmp.shape[0],-1*tmp.shape[1]:] = tmp
			return tmp_image
		else:
			return self.sess.run(self.out,feed_dict={self.input:x})

	def train(self,iterations=10000,save_dir="saved_models"):
		#Removing previous save directory if there is one
		if os.path.exists(save_dir):
			shutil.rmtree(save_dir)
		#Make new save directory
		os.mkdir(save_dir)
		#Just a tf thing, to merge all summaries into one
		merged = tf.summary.merge_all()
		#Using adam optimizer as mentioned in the paper
		optimizer = tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9,beta2=0.999)
		#This is the train operation for our objective
		train_op = optimizer.minimize(self.loss)	
		#Operation to initialize all variables
		init = tf.global_variables_initializer()
		print("Begin training...")
		config = tf.ConfigProto()
		config.gpu_options.allow_growth=True
		config.gpu_options.per_process_gpu_memory_fraction = 0.8
		with self.sess as sess:
			#Initialize all variables
			sess.run(init)
			print("init variables...")
			#create summary writer for train
			train_writer = tf.summary.FileWriter(save_dir+"/train",sess.graph)
			print("save graph...")
			iternum=1
			#This is our training loop
			for i in tqdm(range(iterations)):
				#Use the data function we were passed to get a batch every iteration
				x=[]
				y=[]
				imgx = os.listdir("./data_dir_crop_x/")
				sample = random.sample(imgx, 32)
				for name in sample:
					a, b = os.path.splitext(name)
					x.append(scipy.misc.imread("./data_dir_crop_x/"+a+b))
					y.append(scipy.misc.imread("./data_dir_crop_y/"+a+b))
				#Create feed dictionary for the batch
				feed = {
					self.input:x,
					self.target:y
				}
				#Run the train op and calculate the train summary
				summary,_ = sess.run([merged,train_op],feed)
				train_writer.add_summary(summary,i)
			#Save our trained model		
				if(iternum%100==0):
					print("save model"+str(i))
					self.save()
				iternum = iternum + 1
			self.save()
