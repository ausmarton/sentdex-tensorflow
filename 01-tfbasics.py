import tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(6)

# tf.mul has been replaced with tf.multiply according to:
# https://stackoverflow.com/questions/42217059/tensorflowattributeerror-module-object-has-no-attribute-mul
result = tf.multiply(x1,x2)
print(result)

with tf.Session() as sess:
    output = sess.run(result)
    print(output)
    
print(output)

