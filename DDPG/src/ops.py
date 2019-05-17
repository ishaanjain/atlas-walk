import tensorflow as tf

def fully_connected(inputs,
                    output_size,
                    w_init=tf.glorot_uniform_initializer(),
                    b_init=tf.constant_initializer(0.0),
                    batch_norm=True,
                    is_training=True,
                    scope=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # input_size = inputs.get_shape().as_list()[0]
        input_size = inputs.shape[1]

        weights = tf.get_variable('weights', shape=[input_size, output_size],
                                  dtype=tf.float32, initializer=w_init)

        biases = tf.get_variable('biases', shape=[output_size], dtype=tf.float32,
                                 initializer=b_init)

        dense = tf.matmul(inputs, weights) + biases

        if batch_norm:
            dense = tf.layers.batch_normalization(dense, training=is_training, name='dense_batchnorm')

        return dense


def activation_function(inputs,
         scope=None,
         activation=tf.nn.relu):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        output = activation(inputs)

        return output

#tf.random_uniform_initializer
#mean_square: tf.square(self.predicted_q_value-self.out)
