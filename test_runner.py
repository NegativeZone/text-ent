import tensorflow as tf

hello_constant = tf.constant("Hello from tensorflow!")

def test():
    output = None
    with tf.Session() as sess:
        output = sess.run(hello_constant)
        print(output)
    return output
