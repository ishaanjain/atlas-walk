import tensorflow as tf

def fully_connected(inputs,
                    output_size,
                    batch_norm=True,
                    is_training=True,
                    scope=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        input_size = inputs.get_shape().as_list()[0]

        weights = tf.get_variable('weights', shape=[input_size, output_size],
                                  dtype=tf.float32, initializer=tf.glorot_uniform_initializer())

        biases = tf.get_variable('biases', shape=[output_size], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0))

        dense = tf.matmul(inputs, weights) + biases

        if batch_norm:
            dense = tf.layers.batch_normalization(dense, training=is_training, name='dense_batchnorm')

        return dense


def relu(inputs,
         scope=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        output = tf.nn.relu(inputs)

        return output
