import tensorflow as tf
import DMVST_Net_CNN
import DMVST_Net_LSTM
import DMVST_Net_Semantic

FC_SIZE = 512
OUTPUT_NODE = 1
CONTEXT_NODE = 4


#图片数据、环境数据是都是 list
def forward(x0, x1, x2, x3, x4, x5, x6, x7, e0, e1, e2, e3, e4, e5, e6, e7, m0,
            train, batch_size, regularizer):
    g = []
    x = [x0, x1, x2, x3, x4, x5, x6, x7]
    e = [e0, e1, e2, e3, e4, e5, e6, e7]
    for i in range(DMVST_Net_LSTM.NUM_STEPS):
        s = DMVST_Net_CNN.localcnn(x[i], train, regularizer)
        g.append(tf.concat([s, e[i]], axis=1))
    h = DMVST_Net_LSTM.make_lstm(g, batch_size)
    temp = tf.constant([0.] * DMVST_Net_LSTM.INPUTHIDDEN)
    h1 = tf.add(h,temp,name='mid')

    #之后加入语义视图的输出 m ，q=tf.contact([h, m])
    m = DMVST_Net_Semantic.forward(m0, regularizer, train)
    q = tf.concat([h, m], axis=1)

    # w1 = DMVST_Net_CNN.get_weight([128 + 6, FC_SIZE], regularizer)
    # b1 = DMVST_Net_CNN.get_bias([FC_SIZE])
    # y1 = tf.nn.sigmoid(tf.matmul(q, w1) + b1)
    # if train:
    #     y1 = tf.nn.dropout(y1, 0.5)

    # w2 = DMVST_Net_CNN.get_weight([FC_SIZE, OUTPUT_NODE], regularizer)
    # b2 = DMVST_Net_CNN.get_bias([OUTPUT_NODE])
    # y = tf.nn.sigmoid(tf.matmul(y1, w2) + b2)
    # return y

    w = DMVST_Net_CNN.get_weight([DMVST_Net_LSTM.INPUTHIDDEN + DMVST_Net_Semantic.OUTPUT_NODE, OUTPUT_NODE], regularizer)
    b = DMVST_Net_CNN.get_bias([OUTPUT_NODE])
    q=tf.layers.batch_normalization(q,training=train)
    y = tf.nn.sigmoid(tf.matmul(q, w) + b,name='op_to_store')
    return y
