
# coding: utf-8

# In[16]:


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

def neural_network_model(data, layers = 3, nodes = 500, output_classes = 10):
    _, dimension = data.shape
    input_size = dimension.value
        
    if layers == 0:
        output_layer = create_layer_config(input_size, output_classes)
        return tf.matmul(data, output_layer['weights']) + output_layer['biases']
    else:
        hidden_layer = create_layer_config(input_size, nodes)
        layer = create_hidden_layer(hidden_layer, data)
        return neural_network_model(layer, layers - 1, nodes,output_classes)

def create_hidden_layer(layer_config, inputs):
    layer = tf.add(tf.matmul(inputs, layer_config['weights']), layer_config['biases'])
    return tf.nn.relu(layer)

def create_layer_config(inputs,nodes):
    return {'weights': tf.Variable(tf.random_normal([inputs, nodes])),
                     'biases': tf.Variable(tf.random_normal([nodes]))}

def train_neural_network(x):
    prediction = neural_network_model(x)
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

