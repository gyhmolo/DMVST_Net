import tensorflow as tf

#local_cnn的超参数

#图片分辨率为9*9
IMAGE_SIZE = 9
# mnist数据集为灰度图，输入图片的通道数为 1
NUM_CHANNELS = 1

#共 3 个卷积层
#每层卷积核大小为 3
CONV1_SIZE = 3
#卷积核个数为 32
CONV1_KERNEL_NUM = 64

CONV2_SIZE = 3
CONV2_KERNEL_NUM = 64

CONV3_SIZE = 3
CONV3_KERNEL_NUM = 64

#全连接第一层(隐藏层)为 512 个神经元
FC_SIZE = 512
#作为 LSTM 输入的 CNN 输出维数为64
OUTPUT_NODE = 64


#定义初始化网络权重参数
def get_weight(shape, regularizer):
    # tf.truncated_normal 生成去掉过大偏离点的正态分布随机数的张量，stddev指定标准差
    w = tf.Variable(tf.truncated_normal(shape, mean=0.001, stddev=0.0002))

    #如果使用正则化，则把每一个 w 的正则化记录到总 losses
    if regularizer != None:
        #为权重加入L2正则化，通过限制权重的大小，使模型不会随意拟合训练数据中的随机噪声
        tf.add_to_collection('losses',
                             tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


#定义初始化偏置项函数
def get_bias(shape):
    #将偏置统一初始化为0
    b = tf.Variable(tf.zeros(shape))
    return b


#定义卷积计算函数
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


# 定义最大池操作函数
def max_pool_2x2(x):
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


#定义local-cnn
def localcnn(x, train, regularizer):
    #定义空间视图

    #第一层 cnn
    conv1_w = get_weight(
        [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM], regularizer)
    conv1_b = get_bias([CONV1_KERNEL_NUM])
    conv1 = conv2d(x, conv1_w)
    conv1=tf.layers.batch_normalization(conv1,training=train)
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    pool1 = max_pool_2x2(relu1)
    # relu1_mean, relu1_var = tf.nn.moments(relu1, axes=[0,1])

    #第二层 cnn
    conv2_w = get_weight(
        [CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM],
        regularizer)
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    # conv2 = conv2d(relu1, conv2_w)
    conv2 = conv2d(pool1, conv2_w)
    conv2=tf.layers.batch_normalization(conv2,training=train)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    pool2 = max_pool_2x2(relu2)

    #第三层 cnn
    conv3_w = get_weight(
        [CONV3_SIZE, CONV3_SIZE, CONV2_KERNEL_NUM, CONV3_KERNEL_NUM],
        regularizer)
    conv3_b = get_bias([CONV3_KERNEL_NUM])
    # conv3 = conv2d(relu2, conv3_w)
    conv3 = conv2d(pool2, conv3_w)
    conv3=tf.layers.batch_normalization(conv3,training=train)
    relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_b))
    pool3 = max_pool_2x2(relu3)

    #拉伸 cnn 的输出
    # cnn_shape = relu3.get_shape().as_list()
    cnn_shape = pool3.get_shape().as_list()
    # relu3_shape[0]是一个 batch 值
    #从 list 依次取出矩阵的长宽以及深度，得到矩阵拉伸后的长度
    nodes = cnn_shape[1] * cnn_shape[2] * cnn_shape[3]
    #将 pool2 转换为一个 batch 的向量再传入后续的全连接神经网络
    # reshaped = tf.reshape(relu3, [cnn_shape[0], nodes])
    reshaped = tf.reshape(pool3, [cnn_shape[0], nodes])

    # #第一层 fc
    # fc1_w = get_weight([nodes, FC_SIZE], regularizer)
    # fc1_b = get_bias([FC_SIZE])
    # fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
    # #如果是训练阶段，则对该层输出使用 dropout ，也就是随机地将该层输出的一半神经元置为无效，只用于FC
    # if train:
    #     fc1 = tf.nn.dropout(fc1, 0.5)

    # #第二层 fc
    # fc2_w = get_weight([FC_SIZE, OUTPUT_NODE], regularizer)
    # fc2_b = get_bias([OUTPUT_NODE])
    # s = tf.nn.relu(tf.matmul(fc1, fc2_w) + fc2_b)

    # return fc1
    fc_w = get_weight([nodes,OUTPUT_NODE],regularizer)
    fc_b = get_bias([OUTPUT_NODE])
    s=tf.nn.relu(tf.matmul(reshaped,fc_w)+fc_b)
    return s
