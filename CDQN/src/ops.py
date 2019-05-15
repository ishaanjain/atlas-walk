import tensorflow as tf

def fully_connected(inputs,
                    output_size,
                    w_init=tf.glorot_uniform_initializer(),
                    b_init=tf.constant_initializer(0.0),
                    batch_norm=True,
                    is_training=True,
                    scope=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        input_size = inputs.shape[0]

        weights = tf.get_variable('weights', shape=[input_size, output_size],
                                  dtype=tf.float32, initializer=w_init)

        biases = tf.get_variable('biases', shape=[output_size], dtype=tf.float32,
                                 initializer=b_init)

        dense = tf.matmul(inputs, weights) + biases

        if batch_norm:
            dense = tf.layers.batch_normalization(dense, training=is_training, name='dense_batchnorm')

        return dense


def activation_fn(inputs,
                  fn=tf.nn.relu,
                  scope=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        output = fn(inputs)

        return output


def mean_squared_error(Q,
                       target,
                       scope=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        mse = tf.squared_difference(Q, target)
        loss = tf.reduce_mean(mse, name='loss')

        return loss


def createLowerTriangle(linear_layer,
                        action_size,
                        scope='L'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        l = tf.reshape(linear_layer, (None, action_size, action_size))
        diag = tf.matrix_diag_part(l)
        exponentiated_diag = tf.linalg.set_diag(l, tf.exp(diag))
        lowerTriangle = tf.linalg.band_part(exponentiated_diag, -1, 0)

    return lowerTriangle
