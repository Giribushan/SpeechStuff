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
simpleModel=True

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


def _convert_to_example(filename, image_buffer, label, text, height, width):
	#This method returns a proto

	colorspace = 'RGB'
	channels = 3
	image_format = 'JPEG'

	example = tf.train.Example(features = tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
      'image/channels': _int64_feature(channels),
      'image/class/label': _int64_feature(label),
      'image/class/text': _bytes_feature(tf.compat.as_bytes(text)),
      'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
      'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
      'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}))
	return example


##Function to tell the tensorflow how to read a single image from the input file.
def getImage(filename):
	#convert the filenames to a queue for the input pipeline.
	filenameQ = tf.train.string_input_producer([filename], num_epochs = None)
	#Object to read the records
	recordReader = tf.TFRecordReader()
	#Read the full set of input features from the example..
	key, fullExample = recordReader.read(filenameQ)
	print("The fileName received - " + str(filename))
	print("The String input producer - " + str(filenameQ))
	#print("The full Example - " + str(fullExample.eval(session = sess)))
	tf.Print(input_ = fullExample, data = [fullExample], message = "The full Example read from the RecordReader")
	#tf.Print(fullExample)
	#parse the full example into its component features..
	features = tf.parse_single_example(fullExample,features={'image/height': tf.FixedLenFeature([], tf.int64),'image/width': tf.FixedLenFeature([], tf.int64),'image/colorspace': tf.FixedLenFeature([], dtype=tf.string,default_value=''),'image/channels':  tf.FixedLenFeature([], tf.int64),'image/class/label': tf.FixedLenFeature([],tf.int64),'image/class/text': tf.FixedLenFeature([], dtype=tf.string,default_value=''),'image/format': tf.FixedLenFeature([], dtype=tf.string,default_value=''), 'image/filename': tf.FixedLenFeature([], dtype=tf.string,default_value=''),'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value='')})
    #features = tf.parse_single_example(fullExample,features={'image/height': tf.FixedLenFeature([], tf.int64),'image/width': tf.FixedLenFeature([], tf.int64),'image/colorspace': tf.FixedLenFeature([], dtype=tf.string,default_value=''),'image/channels':  tf.FixedLenFeature([], tf.int64),'image/class/label': tf.FixedLenFeature([],tf.int64),'image/class/text': tf.FixedLenFeature([], dtype=tf.string,default_value=''),'image/format': tf.FixedLenFeature([], dtype=tf.string,default_value=''),'image/filename': tf.FixedLenFeature([], dtype=tf.string,default_value=''),'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value='')})

	#now we have defined the component feature space, for one single example..
	#then we are going to manipulate the labels, and its features, such that we can read each of them into batches..!
	label = features['image/class/label']
	image_buffer = features['image/encoded']

	##Now we need to decode the jpeg image..!!
	##let define the decoding name space..!!
	with tf.name_scope('decode_jpeg', [image_buffer], None):
		#decode
		print("We are decode Operation naming scope..!!")
		image = tf.image.decode_jpeg(image_buffer, channels = 3)
		#convert the string type of the image to single precision datatype..!
		image = tf.image.convert_image_dtype(image, dtype = tf.float32)

		#case the image to singel array, where each element correspinds to greysclae value of a single pixel
		image = tf.reshape(1-tf.image.rgb_to_grayscale(image), [height*width])
		#in the above stmt, 1- will invert the image, and the backgroupd is black

		label = tf.stack(tf.one_hot(label-1, nClass))

		return label, image

##maipulated the image, and label feature in a right format, for training


label, image = getImage("C:\deeplearning\speechStuff\data/train-00001-of-00002")
vlabel, vimage = getImage("C:\deeplearning\speechStuff\data/validation-00000-of-00001")

##get the associated label batch and the IMAGE batch
imageBatch, labelBatch = tf.train.shuffle_batch(
	[image, label], batch_size = 10,
	capacity = 1500,
	min_after_dequeue = 1000)

##similarly for the validation set..
vimageBatch, vlabelBatch = tf.train.shuffle_batch(
	[vimage, vlabel],batch_size = 5,
	capacity = 2000,
	min_after_dequeue = 1000)

##lets create an Interactive session..]
sess  = tf.InteractiveSession()

x= tf.placeholder(tf.float32, [None, width*height])
#plce holder for the true outputs..
y = tf.placeholder(tf.float32, [None, nClass])


## Defining the MOdel here..
if True : 
	print("Running Simple y = mx + b")
	#define the weight tensor, 
	#define the bias tensor, and calculate the y using y = mx+b, then finally applying the softmax layer..!!
else:
	#settup the venilla versions of the conv and pooling layers
	def conv2d(x,W):
		return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding  = 'SAME')
	def max_pool_2x2(x):
		return tf.nn.max_pool(x, ksize = [1,2,2,1],
								 strides = [1,2,2,1],
								 padding  = 'SAME')
	print("runing the conv n/w")
	'''
	do somethig some thign....and calculate the y similar to the if block...
	'''

##Now ready to initialize the and run the input queues and then training..(if needed)
sess.run(tf.global_variables_initializer())

#start the threads for reading the file..
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess = sess, coord = coord)

#start training ..!!
nSteps = 1000
for i in range(1000):
	print("The STEP Number - " + str(i))
	batch_xs, batch_ys = sess.run([imageBatch, labelBatch])
	print(batch_xs)
	print(batch_ys)

	#############################################
	''' STOP here!!!'''

























class ImageCoder(object):
	'''helper class provides the tensorflow coding utilities..'''
	def __init__(self):
		sess = tf.Session()