#其实语义视图的前两部分已经在数据处理的时候完成了，这里只需要搭建语义视图最后的
#全连接神经网络即可
import tensorflow as tf

INPUT_NODE = 32
OUTPUT_NODE = 6
LAYER_NODE = 128


#加入正则化
def get_weight(shape, regularizer):
    #初始化 w ，随机生成参数 w
    w = tf.Variable(tf.truncated_normal(shape, mean=0.001, stddev=0.0002))
    #对 w 进行正则化
    if regularizer != None:
        tf.add_to_collection('losses',
                             tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


#前向传播，搭建神经网络，描述从输入到输出的数据流
def forward(x, regularizer, train):
    # #第一层输入 w1
    # w1 = get_weight([INPUT_NODE, LAYER_NODE], regularizer)
    # #第一层偏置 b1
    # b1 = get_bias([LAYER_NODE])
    # #第一层输出 y1
    # y1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    # if train:
    #     y1 = tf.nn.dropout(y1, 0.5)

    # #第2层输入 w2
    # w2 = get_weight([LAYER_NODE, OUTPUT_NODE], regularizer)
    # #第2偏置 b2
    # b2 = get_bias([OUTPUT_NODE])
    # #第2层输出 y
    # y = tf.matmul(y1, w2) + b2

    # return y
    w = get_weight([INPUT_NODE, OUTPUT_NODE], regularizer)
    b = get_bias([OUTPUT_NODE])
    y = tf.nn.relu(tf.matmul(x, w) + b,name='se')
    return y