import tensorflow as tf

sess = tf.InteractiveSession()
# Create two random matrices
a = tf.Variable(tf.random_normal([4, 5], stddev=2))
b = tf.Variable(tf.random_normal([4, 5], stddev=2))
# Element Wise Multiplication
A = a * b
a.initializer.run()
b.initializer.run()
print(A.eval())
