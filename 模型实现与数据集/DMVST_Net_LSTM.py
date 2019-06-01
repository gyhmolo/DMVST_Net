import numpy as np
import tensorflow as tf
import DMVST_Net_CNN

#定义 LSTM 的超参数
INPUTHIDDEN = 64
# DIMOUTPUT = 10
NUM_STEPS = 8


def make_lstm(current_input, batch_size):
    
    lstm = tf.nn.rnn_cell.BasicLSTMCell(INPUTHIDDEN)
    state = lstm.zero_state(batch_size, tf.float32)
    for i in range(NUM_STEPS):
        with tf.variable_scope("for_reuse_scope"):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            lstm_output, state = lstm(current_input[i], state)

    return lstm_output



# def _RNN(_X, _nsteps, _name, regularizer):

#     w1 = DMVST_Net_CNN.get_weight([diminput, INPUTHIDDEN], regularizer)
#     b1 = DMVST_Net_CNN.get_bias([INPUTHIDDEN])
#     #input layer=>hidden layer
#     _H = tf.matmul(_X, w1) + b1
#     with tf.variable_scope(_name) as scope:
#         scope.reuse_variables()
#         lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(INPUTHIDDEN, forget_bias=1.0)
#         _LSTM_O, _LSTM_S = tf.nn.run(lstm_cell, _H, dtype=tf.float32)
#     _O = _LSTM_O[-1]

#     #使用多层Tensorflow接口将多层的LSTM结构连接成RNN网络并计算前向传播结果
#     cell = tf.nn.rnn_cell.MultiRNNCell(
#         [tf.nn.rnn_cell.BasicLSTMCell(INPUTHIDDEN) for _ in range(NUM_LAYERS)])

#     outputs, _ = tf.nn.dynamic_rnn(cell, )
