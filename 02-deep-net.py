
# coding: utf-8

# In[22]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

batch_size = 100

#height x width
# x = height x width = 784 => one dimensional sequence from a matrix
#input data
x = tf.placeholder('float',[None,784])
#label of the data
y = tf.placeholder('float')

def neural_network_model(data, layer_nodes = [500,500,500], output_classes = 10):
    if layer_nodes == []:
        return create_layer(data, output_classes)
    else:
        # using * to unpack tail into next_layer_config
        # requires Python 3.x
        # https://stackoverflow.com/questions/10532473/python-head-and-tail-in-one-line
        nodes,*next_layer_nodes = layer_nodes
        layer = create_layer(data, nodes)
        return neural_network_model(tf.nn.relu(layer), next_layer_nodes,output_classes)

def create_layer(inputs, nodes):
    _, dimension = inputs.shape
    input_size = dimension.value
    
    weights = tf.Variable(tf.random_normal([input_size, nodes]))
    biases = tf.Variable(tf.random_normal([nodes]))
    
    return tf.add(tf.matmul(inputs, weights), biases)


def train_neural_network(x):
    prediction = neural_network_model(x, layer_nodes = [600, 400, 200])
    # softmax_cross_entropy_with_logits has been deprecated and it does not accept unnamed parameters
    # https://github.com/tensorflow/tensorflow/blob/v1.5.0/tensorflow/python/ops/nn_ops.py#L1833
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 10
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x,epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer,cost], feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
            
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
            
train_neural_network(x)

