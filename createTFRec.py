from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import tensorflow as tf
import sys
import threading
import numpy as np

#number of classes is 2 (squares and triangles)
nClass=2

#simple model (set to True) or convolutional neural network (set to False)
simpleModel=False

#dimensions of image (pixels)
height=32
width=32

tf.app.flags.DEFINE_string('train_directory', '/tmp/',
						   'Training data directory')
tf.app.flags.DEFINE_string('validation_directory', '/tmp',
					       'Validation data directory')
tf.app.flags.DEFINE_string('output_directory', '/tmp/',
                           'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 2,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 2,
                            'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 2,
                            'Number of threads to preprocess the images.')

tf.app.flags.DEFINE_string('labels_file', '', 'Labels file')

def _int64_feature(value):
	''' wrapper or inserting inte64 features into Example proto'''
	if not isinstance(value, list):
		value = [value]
	return tf.train.Feature(int64_list = tf.train.Int64List(value = value))



##Function to tell the tensorflow how to read a single image from the input file.
def getImage(filename):
	#convert the filenames to a queue for the input pipeline.
	filenameQ = tf.train.string_input_producer([filename], num_epochs = None)
	#Initializing the tf Record reader
	recordReader = tf.TFRecordReader()
	#Read the full set of input features from the example..
	key, fullExample = recordReader.read(filenameQ)
	tf.Print(input_ = fullExample, data = [fullExample], message = "The full Example read from the RecordReader")
	#tf.Print(fullExample)
	#parse the full example into its component features..
	##The parse_single_example is retuning the a dict of keys, and tensors.
	## Mainly it is used to read the serialized protos, and convert to tensors for further processing.
	features = tf.parse_single_example(fullExample,features={'image/height': tf.FixedLenFeature([], tf.int64),'image/width': tf.FixedLenFeature([], tf.int64),'image/colorspace': tf.FixedLenFeature([], dtype=tf.string,default_value=''),'image/channels':  tf.FixedLenFeature([], tf.int64),'image/class/label': tf.FixedLenFeature([],tf.int64),'image/class/text': tf.FixedLenFeature([], dtype=tf.string,default_value=''),'image/format': tf.FixedLenFeature([], dtype=tf.string,default_value=''), 'image/filename': tf.FixedLenFeature([], dtype=tf.string,default_value=''),'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value='')})
	#So now lets access the dict of tensors using their keys..!!
	#now we have defined the component feature space, for one single example..
	#then we are going to manipulate the labels, and its features, such that we can read each of them into batches..!
	label = features['image/class/label']
	image_buffer = features['image/encoded']

	##Now we need to decode the jpeg image..!!
	##let define the decoding name space..!!
	with tf.name_scope('decode_jpeg', [image_buffer], None):
		#decode
		print("Entered the decoding operation naming scope..!!")
		image = tf.image.decode_jpeg(image_buffer, channels = 3)
		#convert the string type of the image to single precision datatype..!
		image = tf.image.convert_image_dtype(image, dtype = tf.float32)

		#case the image to singel array, where each element correspinds to greysclae value of a single pixel
		################### ???????????????????????????????????????????????
		image = tf.reshape(1-tf.image.rgb_to_grayscale(image), [height*width])
		#image = tf.reshape(1-tf.image.hsv_to_rgb(image),[height*width])
		#mage = 1-tf.image.hsv_to_rgb(image)
		#image = tf.reshape(image, [height*width])
		#in the above stmt, 1- will invert the image, and the backgroupd is black

		label = tf.stack(tf.one_hot(label-1, nClass))
		print("The image and label shapes...")
		print(image.shape)
		print(label.shape)

		return label, image

##maipulated the image, and label feature in a right format, for training


label, image = getImage("C:\deeplearning\speechStuff\data/train-00001-of-00002")
vlabel, vimage = getImage("C:\deeplearning\speechStuff\data/validation-00000-of-00001")


print("The following are the operations...we are just declaring but net yet running..")
##get the associated label batch and the IMAGE batch
imageBatch, labelBatch = tf.train.shuffle_batch(
	[image, label], batch_size = 100,
	capacity = 2000,
	min_after_dequeue = 1000)

##similarly for the validation set..
vimageBatch, vlabelBatch = tf.train.shuffle_batch(
	[vimage, vlabel],batch_size = 100,
	capacity = 2000,
	min_after_dequeue = 1000)

##lets create an Interactive session..]
#sess  = tf.InteractiveSession()

x= tf.placeholder(tf.float32, [None, width*height])
#plce holder for the true outputs..
y_ = tf.placeholder(tf.float32, [None, nClass])


## Defining the MOdel here..
if simpleModel : 
	print("Running Simple y = mx + b")
	#In simple regression model, W maps the input to output so its size should be (num of pixels) * (nClasses)
	W = tf.Variable(tf.zeros([width * height, nClass]))
	b = tf.Variable(tf.zeros([nClass]))

	## Now lets define the softmax function.>!!
	y = tf.nn.softmax(tf.matmul(x, W) + b)
	cross_entropy =	tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))

	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	correct_prediction  = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	print("Crossd the correction prediction step")
	print(type(correct_prediction))
	print()
	correct_prediction  = tf.cast(correct_prediction, tf.float32)
	#Get the mean of all entries in correct prediction, The Higher the better
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

else:
	#Lets randomize the initial weights and biases 
	def weight_variable(shape):
		print("The shape passed to the Weight operations - ")
		print(shape)
		shape = np.int64(shape)
		initial = tf.truncated_normal(shape, stddev = 0.1)
		return tf.Variable(initial)
	def bias_variable(shape):
		initial = tf.constant(0.1, shape = shape)
		return tf.Variable(initial)
	## NOW setup the Vanilla version of convolution and pooling..
	def conv2d(x,W):
		return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding  = 'SAME')
	# For maxpooing the input shoud be a 4D-tensor, with [batch, height, width, channels], and the type as tf.float32
	def max_pool_2x2(x):
		return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding  = 'SAME')
	print("runing the conv n/w")
	# we are setting up 2 conv layers and one fully connected layer. 
	# Below are the corresponding feature size
	nFeatures1 = 32
	nFeatures2 = 64
	nNeuronsfc  = 1024

	#Use the above functions to initialize the weights, and biases.
	#Filter / weight size ( 5x5)
	#1 input channel - since it is a gray scale input image.
	#[filter_height * filter_width * in_channels, output_channels]
	W_conv1 = weight_variable([5,5,1,nFeatures1])
	b_conv1 = bias_variable([nFeatures1])

	## Now let reshape the raw image into 4d tensor
	## Because the conv2d layer accepts the images with only following dimensions..
	## [batch, height, width, channels]
	## So lets reshape it..keeping the width, height, and num_channels constant, and expand the other dimension.
	x_image = tf.reshape(x, [-1, width, height, 1])


	##Hidden layer1 
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	##lets apply pooling...it will reduce each dimension by factor - 2
	h_pool1 = max_pool_2x2(h_conv1)
	#casting 
	nFeatures1 = np.int64(nFeatures1)
	##Similar to the above operations...for conv layer2
	W_conv2 = weight_variable([5,5,nFeatures1, nFeatures2])
	b_conv2 = bias_variable([nFeatures2])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)


	# densely connected layer. Similar to above, but operating
	# on entire image (rather than patch) which has been reduced by a factor of 4 
	# in each dimension
	# so use large number of neurons 

	#check our dimensions are multiple of 4...because the input image is 32x32 resolution
	if(width%4 or height%4):
		print("Error: width and height must be multiple of 4")
		sys.exit(1)
	nFeatures2 = np.int64(nFeatures2)
	nNeuronsfc = np.int64(nNeuronsfc)
	print("width - " + str(width))
	print('height - ' + str(height))
	#W_fc1 = weight_variable([tf.cast(((width/4) * (height/4)), dtype = tf.int64) * nFeatures2 , nNeuronsfc])
	height = np.int64(height)
	width = np.int64(width)
	W_fc1 = weight_variable([(width/4) * (height/4) * 64 , 1024])
	print("The weights of fully connected layers.. " + str(W_fc1))

	b_fc1 = bias_variable([nNeuronsfc])
	print("The biases of fully connected layer .. " + str(b_fc1))

	##This is where you have squashed the 4d to 1d and flattened
	h_pool2_flat = tf.reshape(h_pool2, np.int64([-1, (width/4) * (height/4) * 64]))
	#Then applying the non linearity
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1 )

	###Then reduce the overfitting by applying the dropout 
	### SO the each neuron kept with the probability of Keepprob.
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob = keep_prob)

	#create readout layer which outputs to nClass category
	W_fc2 = weight_variable([nNeuronsfc, nClass])
	b_fc2 = bias_variable([nClass])

	##Now define the output calculation(Softmax), this give the 
	##probability distributions for all output classes.!
	y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
	print("These are the logits COMPUTED..." + str(y))

	### Now before start training we need to define the error and cost terms..!!
	#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = y_, labels = y)
	cross_entropy =	tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))

	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	correct_prediction  = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	print("Crossd the correction prediction step")
	print(type(correct_prediction))
	print()
	correct_prediction  = tf.cast(correct_prediction, tf.float32)
	#Get the mean of all entries in correct prediction, The Higher the better
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


## Now let try creating a session here...
sess  = tf.InteractiveSession()
##Now lets initiate the variables
print("Initialing the global variables...")
sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
print("Starting the queue runners...")
threads = tf.train.start_queue_runners(sess = sess, coord = coord)

##Now lets start training
nSteps = 1000
print("Fixed the number of steps - " + str(nSteps) + ' --  and start TRAINING..!!')
for i in range(nSteps):
	print("Reading the as batches..")
	batch_xs, batch_ys = sess.run([imageBatch, labelBatch])
	##Choose which model you wants to train.
	if simpleModel:
		train_step.run(feed_dict = {x:batch_xs, y_ : batch_ys})
	else:
		#Running the training step
		train_step.run(feed_dict = {x:batch_xs, y_ : batch_ys, keep_prob :0.5})


	##Then perform validation..
	if (i+1) % 10 == 0: ## for every 2 steps!!
		#similar process in getting the validation batches..
		vbatch_xs, vbatch_ys = sess.run([vimageBatch, vlabelBatch])
		if simpleModel:
			train_accuracy = accuracy.eval(feed_dict = {x:vbatch_xs, y_ : batch_ys})
			#print("The predicted labels.. " +str(sess.run(y_)))
			#print("The actual labels.. " +str(sess.run(y)))
		else:
			train_accuracy = accuracy.eval(feed_dict = {x:vbatch_xs, y_ : batch_ys, keep_prob : 1.0})
			#print("The predicted labels.. " +str(sess.run(y_, feed_dict = {y_ : batch_ys})))
			#print("The actual labels.. " +str(sess.run(y, feed_dict = { y : [label]})))
		print("Step %d, training accuracy %g"%(i+1, train_accuracy ))

if not simpleModel:
	print("Time to wrap up the training process - close the all the Queue runners..!")
	coord.request_stop()
	coord.join(threads)

class ImageCoder(object):
	'''helper class provides the tensorflow coding utilities..'''
	def __init__(self):
		sess = tf.Session()
