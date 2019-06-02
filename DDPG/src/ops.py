import tensorflow as tf

def fully_connected(inputs,
                    output_size,
                    w_init=tf.initializers.lecun_normal(),
                    b_init=tf.initializers.lecun_normal(),
                    use_bias=True,
                    batch_norm=False,
                    is_training=True,
                    scope=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        input_size = inputs.get_shape().as_list()[1]

        weights = tf.get_variable('weights', shape=[input_size, output_size],
                                  dtype=tf.float32, initializer=w_init)

        if (use_bias):
            biases = tf.get_variable('biases', shape=[output_size], dtype=tf.float32,
                                    initializer=b_init)

            dense = tf.matmul(inputs, weights) + biases
        else:
            dense = tf.matmul(inputs, weights)

        if batch_norm:
            mean, variance = tf.nn.moments(dense, [0])
            scale = tf.get_variable('scale', shape=[output_size], dtype=tf.float32,
                                    initializer=tf.constant_initializer(1.0))
            beta = tf.get_variable('beta', shape=[output_size], dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.0))
            dense = tf.nn.batch_normalization(dense, mean, variance, beta, scale, 1e-3, name='batch_norm')

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
