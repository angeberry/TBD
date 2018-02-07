import tensorflow as tf

def mnist_net(x, logits=False, training=False, reuse=True, name='model', size='small'):
    if size == 'small':
        conv_filters = [32, 64]
    elif size == 'large':
        conv_filters = [32, 64, 64, 128]
    else:
        conv_filtesr = [128, 256]
    layers = len(conv_filters)
    with tf.variable_scope(name) as scope:
        if reuse==True:
            scope.reuse_variables()
        z = x
        for layer in range(layers):
            with tf.variable_scope('conv%s'%layer):
                z = tf.layers.conv2d(z, filters=conv_filters[layer], kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
                z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)
        with tf.variable_scope('flatten'):
            z = tf.layers.flatten(z)

            with tf.variable_scope('mlp'):
                if size == 'small':
                    z = tf.layers.dense(z, units=128, activation=tf.nn.relu)
                else:
                    z = tf.layers.dense(z, units=256, activation=tf.nn.relu)
                z = tf.layers.dropout(z, rate=0.25, training=training)

            logits_ = tf.layers.dense(z, units=10, name='logits')
            y = tf.nn.softmax(logits_, name='ybar')
        if logits:
            return y, logits_
        return y

def residual_network(inp, stack_n = 4, logits=False, training=False, reuse=True, name='res_model', n_classes=10):

    def _res_block(input, out_channel, increase=False, training=training):
        if increase:
            stride=[2, 2]
        else:
            stride=[1, 1]
        x = tf.contrib.layers.batch_norm(input, is_training=training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, filters=out_channel, kernel_size=[3, 3], strides=stride,padding='same', activation=None)
        x = tf.contrib.layers.batch_norm(x, is_training=training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, filters=out_channel, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=None)
        if increase:
            projection = tf.layers.conv2d(input, filters=out_channel, kernel_size=[1, 1], strides=[2, 2], padding='same', activation=None)
            block = x + projection
        else:
            block = x + input
        return block
    with tf.variable_scope(name) as scope:
        if reuse == True:
            scope.reuse_variables()
        x = tf.layers.conv2d(inp, filters=16, kernel_size=[3, 3], padding='same')
        for _ in range(stack_n):
            x = _res_block(x, 16)
        x = _res_block(x, 32, increase=True)
        for _ in range(stack_n):
            x = _res_block(x, 32)
        x = _res_block(x, 64, increase=True)
        for _ in range(stack_n):
            x = _res_block(x, 64)

        x = tf.contrib.layers.batch_norm(x, is_training=training)
        x = tf.nn.relu(x)
        x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=2)

        x = tf.layers.flatten(x)
        logits_ = tf.layers.dense(x, units=n_classes, name='logits')
        y = tf.nn.softmax(logits_, name='ybar')
    if logits:
        return y, logits_
    return y
