from __future__ import print_function
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1
dropout_rate = 0.8  # change and check whether how results change

# Network Parameters
n_hidden_1 = 256  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons
n_hidden_3 = 256  # 3rd layer number of neurons

n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Weights and Biases

weight_hidden = tf.Variable(tf.random_normal([n_input, n_hidden_1]))
weight_out = tf.Variable(tf.random_normal([n_hidden_3, n_classes]))

bias_hidden = tf.Variable(tf.random_normal([n_hidden_1]))
bias_out = tf.Variable(tf.random_normal([n_classes]))


# Create model
def multilayer_perceptron(x):

    # Hidden fully connected layer 1 with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weight_hidden), bias_hidden)
    # Dropout applying
    layer_1_out = tf.nn.dropout(layer_1, keep_prob=dropout_rate)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_1_out, weight_out) + bias_out

    return out_layer


# Construct model
predictions = multilayer_perceptron(X)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=Y))

# Loss function using L2 Regularization
l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.01, scope=None)
regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weight_hidden)

regularized_loss = loss + regularization_penalty

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss], feed_dict={X: batch_x, Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    pred = tf.nn.softmax(predictions)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
