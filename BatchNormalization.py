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
epsilon = 1e-3

# Network Parameters
n_hidden_1 = 256  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons
n_hidden_3 = 256  # 3rd layer number of neurons
n_hidden_4 = 256  # 4th layer number of neurons

n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
	'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
	'b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create model
# Batch normalization is applied
def multilayer_perceptron(x):

    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Batch normalization
    batch_mean_1, batch_var_1 = tf.nn.moments(layer_1, axes=[0])
    scale_1 = tf.Variable(tf.ones([batch_size, 256]))
    beta_1 = tf.Variable(tf.zeros([batch_size, 256]))
    layer_1_BN = tf.nn.batch_normalization(layer_1, batch_mean_1, batch_var_1, beta_1, scale_1, epsilon)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1_BN, weights['h2']), biases['b2'])
    # Batch normalization
    batch_mean_2, batch_var_2 = tf.nn.moments(layer_2, axes=[0])
    scale_2 = tf.Variable(tf.ones([batch_size, 256]))
    beta_2 = tf.Variable(tf.zeros([batch_size, 256]))
    layer_2_BN = tf.nn.batch_normalization(layer_2, batch_mean_2, batch_var_2, beta_2, scale_2, epsilon)
    # Hidden fully connected layer with 256 neurons
    layer_3 = tf.add(tf.matmul(layer_2_BN, weights['h3']), biases['b3'])
    # Batch normalization
    batch_mean_3, batch_var_3 = tf.nn.moments(layer_3, axes=[0])
    scale_3 = tf.Variable(tf.ones([batch_size, 256]))
    beta_3 = tf.Variable(tf.zeros([batch_size, 256]))
    layer_3_BN = tf.nn.batch_normalization(layer_3, batch_mean_3, batch_var_3, beta_3, scale_3, epsilon)
    # Hidden fully connected layer with 256 neurons
    layer_4 = tf.add(tf.matmul(layer_3_BN, weights['h4']), biases['b4'])
    # Batch normalization
    batch_mean_4, batch_var_4 = tf.nn.moments(layer_4, axes=[0])
    scale_4 = tf.Variable(tf.ones([batch_size, 256]))
    beta_4 = tf.Variable(tf.zeros([batch_size, 256]))
    layer_4_BN = tf.nn.batch_normalization(layer_4, batch_mean_4, batch_var_4, beta_4, scale_4, epsilon)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_4_BN, weights['out']) + biases['out']

    return out_layer


# Construct model
predictions = multilayer_perceptron(X)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=Y))
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
    print("Accuracy:", accuracy.eval({X: mnist.test.images[:batch_size], Y: mnist.test.labels[:batch_size]}))
