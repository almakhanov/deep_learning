import tensorflow as tf

sess = tf.InteractiveSession()

X = tf.Variable(tf.eye(10))
X.initializer.run()
print(X.eval())

print("\n\n")
A = tf.Variable(tf.random_normal([5, 10]))
A.initializer.run()
# Multiply two matrices
product = tf.matmul(A, X)
print(product.eval())

print("\n\n")
# create a random matrix of 1s and 0s, size 5x10
b = tf.Variable(tf.random_uniform([5, 10], 0, 2, dtype=tf.int32))
b.initializer.run()
print(b.eval())

b_new = tf.cast(b, dtype=tf.float32)
print("\n\n")
print(b_new.eval())
# Cast to float32 data type

# Add the two matrices
t_sum = tf.add(product, b_new)
t_sub = product - b_new
print("A*X _b\n", t_sum.eval())
print("A*X - b\n", t_sub.eval())
