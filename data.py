import scipy.misc
import random
import numpy as np
import os
import tensorflow as tf

train_set = []
test_set = []
batch_index = 0

"""
Load set of images in a directory.
This will automatically allocate a 
random 20% of the images as a test set
data_dir: path to directory containing images
"""
def load_dataset(data_dir, img_size):
	"""img_files = os.listdir(data_dir)
	test_size = int(len(img_files)*0.2)
	test_indices = random.sample(range(len(img_files)),test_size)
	for i in range(len(img_files)):
		#img = scipy.misc.imread(data_dir+img_files[i])
		if i in test_indices:
			test_set.append(data_dir+"/"+img_files[i])
		else:
			train_set.append(data_dir+"/"+img_files[i])
	return"""
	print("f=load_dataset")
	global train_set
	global test_set
	"""imgs = []
	img_files = os.listdir(data_dir)
	for img in img_files:
		try:
			tmp= scipy.misc.imread(data_dir+"/"+img)
			x,y,z = tmp.shape
			coords_x = int(x / img_size)
			coords_y = int(y/img_size)
			coords = [ (q,r) for q in range(coords_x) for r in range(coords_y) ]
			for coord in coords:
				imgs.append((data_dir+"/"+img,coord))
		except:
			print "oops"
	test_size = max(10,int( len(imgs)*0.2))
	print ("data_dir: "+str(len(img_files))+"   crop_to: "+str(len(imgs)))
	random.shuffle(imgs)
	test_set = imgs[:test_size]
	train_set = imgs[test_size:]
	print ("test: "+str(len(test_set))+"   train: "+str(len(train_set)))"""
	return
def data_generator():
	data_path="./data_dir"
	filelist = os.listdir(data_path)
	input_files = [os.path.join(data_path, 'data_dir_crop_x', f) for f in filelist if f.endswith('.jpg')]
	output_files = [os.path.join(data_path, 'data_dir_crop_y', f) for f in filelist if f.endswith('.jpg')]
	input_queue, output_queue = tf.train.slice_input_producer([input_files, output_files],shuffle=True,seed=100,num_epochs=32)
	input_reader = tf.read_file(input_queue)
	output_reader = tf.read_file(output_queue)
	input = tf.image.decode_jpeg(input_reader, channels=3)
	output = tf.image.decode_jpeg(output_reader, channels=3)
	samples = {}
	samples['input']  = input
	samples['output'] = output
	samples = tf.train.shuffle_batch(samples,batch_size=32,num_threads=10,capacity=256,min_after_dequeue=128)
	return samples
"""
Get test set from the loaded dataset
size (optional): if this argument is chosen,
each element of the test set will be cropped
to the first (size x size) pixels in the image.
returns the test set of your data
"""
def get_test_set(original_size,shrunk_size):
	"""for i in range(len(test_set)):
		img = scipy.misc.imread(test_set[i])
		if img.shape:
			img = crop_center(img,original_size,original_size)		
			x_img = scipy.misc.imresize(img,(shrunk_size,shrunk_size))
			y_imgs.append(img)
			x_imgs.append(x_img)"""
	print("f=get_test_set")
	imgs = test_set
	get_image(imgs[0],original_size)
	x = [scipy.misc.imresize(get_image(q,original_size),(shrunk_size,shrunk_size)) for q in imgs]#scipy.misc.imread(q[0])[q[1][0]*original_size:(q[1][0]+1)*original_size,q[1][1]*original_size:(q[1][1]+1)*original_size].resize(shrunk_size,shrunk_size) for q in imgs]
	y = [get_image(q,original_size) for q in imgs]#scipy.misc.imread(q[0])[q[1][0]*original_size:(q[1][0]+1)*original_size,q[1][1]*original_size:(q[1][1]+1)*original_size] for q in imgs]
	return x,y

def get_image(imgtuple,size):
	img = scipy.misc.imread(imgtuple[0])
	x,y = imgtuple[1]
	img = img[x*size:(x+1)*size,y*size:(y+1)*size]
	return img
	

"""
Get a batch of images from the training
set of images.
batch_size: size of the batch
original_size: size for target images
shrunk_size: size for shrunk images
returns x,y where:
	-x is the input set of shape [-1,shrunk_size,shrunk_size,channels]
	-y is the target set of shape [-1,original_size,original_size,channels]
"""
def get_batch(batch_size,original_size,shrunk_size):
	global batch_index##init to zero
	print ("want to a batch")
	"""img_indices = random.sample(range(len(train_set)),batch_size)
	for i in range(len(img_indices)):
		index = img_indices[i]
		img = scipy.misc.imread(train_set[index])
		if img.shape:
			img = crop_center(img,original_size,original_size)
			x_img = scipy.misc.imresize(img,(shrunk_size,shrunk_size))
			x.append(x_img)
			y.append(img)"""
	max_counter = int(len(train_set)/batch_size)##total batch numer
	counter = batch_index % max_counter##start with zero ,if batch_index > max_counter, then loop to feed
	window = [x for x in range(counter*batch_size,(counter+1)*batch_size)]## index of the minibatch,ep: 0-32,33-64...
	imgs = [train_set[q] for q in window]##according index(in window) to get a tuple in train_set.ep:(data_dir/a.jpg,(0,1))
	x = [scipy.misc.imresize(get_image(q,original_size),(shrunk_size,shrunk_size)) for q in imgs]## x downsample imgs
	y = [get_image(q,original_size) for q in imgs]##y original imgs 
	batch_index = (batch_index+1)%max_counter##batch_index grow
	print ("get a batch")
	return x,y

"""
Simple method to crop center of image
img: image to crop
cropx: width of crop
cropy: height of crop
returns cropped image
"""
def crop_center(img,cropx,cropy):
	y,x,_ = img.shape
	startx = random.sample(range(x-cropx-1),1)[0]#x//2-(cropx//2)
	starty = random.sample(range(y-cropy-1),1)[0]#y//2-(cropy//2)
	return img[starty:starty+cropy,startx:startx+cropx]