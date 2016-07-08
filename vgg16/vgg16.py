import tensorflow as tf

# Implementing CNN part of VGG based on http://arxiv.org/pdf/1409.1556v6.pdf

def get_weights(shape):
    return tf.get_variable('weights', shape, initializer = tf.random_normal_initializer(stddev=0.01))
def get_biases(shape):
    return tf.get_variable('biases', shape, initializer = tf.zeros_initializer)

class VGG16:
    VGG_MEAN = [123.68, 116.779, 103.939] # In RGB, not BGR

    def get_all_variables(self):
        raise NotImplementedError

    def build(self, X):
        raise NotImplementedError

# Model D (16 layers)
class VGG16D(VGG16):
    def get_all_variables(self):
        variables = []
        with tf.variable_scope('VGG16D', reuse = True):
            with tf.variable_scope('layer1'):
                variables.append(get_weights([3, 3, 3, 64]))
                variables.append(get_biases([64]))
            with tf.variable_scope('layer2'):
                variables.append(get_weights([3, 3, 64, 64]))
                variables.append(get_biases([64]))
            with tf.variable_scope('layer3'):
                variables.append(get_weights([3, 3, 64, 128]))
                variables.append(get_biases([128]))
            with tf.variable_scope('layer4'):
                variables.append(get_weights([3, 3, 128, 128]))
                variables.append(get_biases([128]))
            with tf.variable_scope('layer5'):
                variables.append(get_weights([3, 3, 128, 256]))
                variables.append(get_biases([256]))
            with tf.variable_scope('layer6'):
                variables.append(get_weights([3, 3, 256, 256]))
                variables.append(get_biases([256]))
            with tf.variable_scope('layer7'):
                variables.append(get_weights([3, 3, 256, 256]))
                variables.append(get_biases([256]))
            with tf.variable_scope('layer8'):
                variables.append(get_weights([3, 3, 256, 512]))
                variables.append(get_biases([512]))
            with tf.variable_scope('layer9'):
                variables.append(get_weights([3, 3, 512, 512]))
                variables.append(get_biases([512]))
            with tf.variable_scope('layer10'):
                variables.append(get_weights([3, 3, 512, 512]))
                variables.append(get_biases([512]))
            with tf.variable_scope('layer11'):
                variables.append(get_weights([3, 3, 512, 512]))
                variables.append(get_biases([512]))
            with tf.variable_scope('layer12'):
                variables.append(get_weights([3, 3, 512, 512]))
                variables.append(get_biases([512]))
            with tf.variable_scope('layer13'):
                variables.append(get_weights([3, 3, 512, 512]))
                variables.append(get_biases([512]))

        return variables

    def build(self, X):
        input_data = tf.sub(X, VGG16.VGG_MEAN)

        with tf.variable_scope('VGG16D'):
            # First, two conv3-64
            with tf.variable_scope('layer1'): # Layer 1, 3x3 depth 64
                l1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_data, get_weights([3, 3, 3, 64]), strides = [1, 1, 1, 1], padding = 'SAME'),
                                get_biases([64])))
            with tf.variable_scope('layer2'): # Layer 2, 3x3 depth 64
                l2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l1, get_weights([3, 3, 64, 64]), strides = [1, 1, 1, 1], padding = 'SAME'),
                                get_biases([64])))

            # Maxpooling
            m2_3 = tf.nn.max_pool(l2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

            # Second, two conv3-128
            with tf.variable_scope('layer3'): # Layer 3, 3x3 depth 128
                l3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(m2_3, get_weights([3, 3, 64, 128]), strides = [1, 1, 1, 1], padding = 'SAME'),
                                get_biases([128])))
            with tf.variable_scope('layer4'): # Layer 4, 3x3 depth 128
                l4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l3, get_weights([3, 3, 128, 128]), strides = [1, 1, 1, 1], padding = 'SAME'),
                                get_biases([128])))

            # Maxpooling
            m4_5 = tf.nn.max_pool(l4, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

            # Third, three conv3-256
            with tf.variable_scope('layer5'): # Layer 5, 3x3 depth 256
                l5 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(m4_5, get_weights([3, 3, 128, 256]), strides = [1, 1, 1, 1], padding = 'SAME'),
                                get_biases([256])))
            with tf.variable_scope('layer6'): # Layer 6, 3x3 depth 256
                l6 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l5, get_weights([3, 3, 256, 256]), strides = [1, 1, 1, 1], padding = 'SAME'),
                                get_biases([256])))
            with tf.variable_scope('layer7'): # Layer 7, 3x3 depth 256
                l7 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l6, get_weights([3, 3, 256, 256]), strides = [1, 1, 1, 1], padding = 'SAME'),
                                get_biases([256])))

            # Maxpooling
            m7_8 = tf.nn.max_pool(l7, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

            # Fourth, three conv3-512
            with tf.variable_scope('layer8'): # Layer 8, 3x3 depth 512
                l8 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(m7_8, get_weights([3, 3, 256, 512]), strides = [1, 1, 1, 1], padding = 'SAME'),
                                get_biases([512])))
            with tf.variable_scope('layer9'): # Layer 9, 3x3 depth 512
                l9 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l8, get_weights([3, 3, 512, 512]), strides = [1, 1, 1, 1], padding = 'SAME'),
                                get_biases([512])))
            with tf.variable_scope('layer10'): # Layer 10, 3x3 depth 512
                l10 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l9, get_weights([3, 3, 512, 512]), strides = [1, 1, 1, 1], padding = 'SAME'),
                                get_biases([512])))
            # Maxpooling
            m10_11 = tf.nn.max_pool(l10, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

            # Fifth, three conv3-512
            with tf.variable_scope('layer11'): # Layer 11, 3x3 depth 512
                l11 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(m10_11, get_weights([3, 3, 512, 512]), strides = [1, 1, 1, 1], padding = 'SAME'),
                                get_biases([512])))
            with tf.variable_scope('layer12'): # Layer 12, 3x3 depth 512
                l12 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l11, get_weights([3, 3, 512, 512]), strides = [1, 1, 1, 1], padding = 'SAME'),
                                get_biases([512])))
            with tf.variable_scope('layer13'): # Layer 13, 3x3 depth 512
                l13 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l12, get_weights([3, 3, 512, 512]), strides = [1, 1, 1, 1], padding = 'SAME'),
                                get_biases([512])))

            return l13

# # Model A (11 layers)
# def VGG16A(X):
#     with tf.variable_scope('VGG16A'):
#         # First, one conv3-64
#         with tf.variable_scope('layer1'): # Layer 1, 3x3 depth 64
#             l1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(X, get_weights([3, 3, 3, 64]), strides = [1, 1, 1, 1], padding = 'SAME'),
#                             get_biases([64])))
#
#         # Maxpooling
#         m1_2 = tf.nn.max_pool(l1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
#
#         # Second, one conv3-128
#         with tf.variable_scope('layer2'): # Layer 2, 3x3 depth 128
#             l2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(m1_2, get_weights([3, 3, 64, 128]), strides = [1, 1, 1, 1], padding = 'SAME'),
#                             get_biases([128])))
#
#         # Maxpooling
#         m2_3 = tf.nn.max_pool(l2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
#
#         # Third, two conv3-256
#         with tf.variable_scope('layer3'): # Layer 3, 3x3 depth 256
#             l3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(m2_3, get_weights([3, 3, 128, 256]), strides = [1, 1, 1, 1], padding = 'SAME'),
#                             get_biases([256])))
#         with tf.variable_scope('layer4'): # Layer 4, 3x3 depth 256
#             l4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l3, get_weights([3, 3, 256, 256]), strides = [1, 1, 1, 1], padding = 'SAME'),
#                             get_biases([256])))
#
#         # Maxpooling
#         m4_5 = tf.nn.max_pool(l4, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
#
#         # Fourth, two conv3-512
#         with tf.variable_scope('layer5'): # Layer 5, 3x3 depth 512
#             l5 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(m4_5, get_weights([3, 3, 256, 512]), strides = [1, 1, 1, 1], padding = 'SAME'),
#                             get_biases([512])))
#         with tf.variable_scope('layer6'): # Layer 6, 3x3 depth 512
#             l6 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l5, get_weights([3, 3, 512, 512]), strides = [1, 1, 1, 1], padding = 'SAME'),
#                             get_biases([512])))
#         # Maxpooling
#         m6_7 = tf.nn.max_pool(l6, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
#
#         # Fifth, two conv3-512
#         with tf.variable_scope('layer7'): # Layer 7, 3x3 depth 512
#             l7 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(m6_7, get_weights([3, 3, 512, 512]), strides = [1, 1, 1, 1], padding = 'SAME'),
#                             get_biases([512])))
#         with tf.variable_scope('layer8'): # Layer 8, 3x3 depth 512
#             l8 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l7, get_weights([3, 3, 512, 512]), strides = [1, 1, 1, 1], padding = 'SAME'),
#                             get_biases([512])))
#
#         return l8
