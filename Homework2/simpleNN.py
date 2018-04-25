import tensorflow as tf
import numpy as np
learning_rate = 1
x_data = [[0, 0], [1, 0], [0, 1], [-1, 0], [1, -1]]
y_data = [[1], [1], [0], [0], [0]]

# placeholders for a tensor that will be always fed.
x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

W1 = tf.Variable(tf.random_normal([2, 2]), name='weight1')
b1 = tf.Variable(tf.random_normal([2]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([2, 1]), name='weight2')
b2 = tf.Variable(tf.random_normal([1]), name='bias2')
hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
   # Initialize TensorFlow variables
   sess.run(tf.global_variables_initializer())

   for step in range(40001):
       cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
       if step % 200 == 0:
           print(step, cost_val)

   # Accuracy report
   h, c, a, weight1, bias1, weight2, bias2 = sess.run([hypothesis, predicted, accuracy, W1, b1, W2, b2],
                      feed_dict={X: x_data, Y: y_data})
   print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
   print("\nW1: ", weight1, "\nB1: ", bias1)
   print("\nW2: ", weight2, "\nB2: ", bias2)
