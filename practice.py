import tensorflow as tf 

x = tf.constant([5,3,2])
y = tf.constant([1,3,4])
z = tf.constant([4,6,4])
a = tf.stack([x,y,z])
b = tf.stack([x,y,z], axis = 1)

def getImage


init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)
print(sess.run(a))
print(sess.run(b))
sess.close()
