import tensorflow as tf
import codecs
import os
import numpy as np
import DMVST_Net_CNN
import DMVST_Net_forward
import DMVST_Net_Semantic
import math

verify_file='./verifying_resault1'
record_file='./records1'

BATCH_SIZE = 64
if os.path.exists(verify_file):
    with codecs.open(verify_file, 'r','utf-8') as read:
        content = [s.strip() for s in read.readlines()]
    LEARNING_RATE_BASE = float(content[-3].split()[-1])
else:
    LEARNING_RATE_BASE=0.0001
LEARNING_RATE_DECAY = 0.99
REGULIZER = 0.0001
GARMA = 0.05
TRAIN_STEPS = 4320
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = './model/'
MODEL_NAME = "DMVST_Net"

#列举输入文件
train_files = tf.train.match_filenames_once(
    './data/train_data/train.tfrecords-loc*')
verify_files = tf.train.match_filenames_once(
    './data/verify_data/verify.tfrecords-loc*')


#定义parser方法从TFRecords中解析数据
def parser(record, pattern):
    features = tf.parse_single_example(
        record,
        features={
            pattern + '_img0': tf.FixedLenFeature([81], tf.float32),
            pattern + '_img1': tf.FixedLenFeature([81], tf.float32),
            pattern + '_img2': tf.FixedLenFeature([81], tf.float32),
            pattern + '_img3': tf.FixedLenFeature([81], tf.float32),
            pattern + '_img4': tf.FixedLenFeature([81], tf.float32),
            pattern + '_img5': tf.FixedLenFeature([81], tf.float32),
            pattern + '_img6': tf.FixedLenFeature([81], tf.float32),
            pattern + '_img7': tf.FixedLenFeature([81], tf.float32),
            pattern + '_context0': tf.FixedLenFeature([4], tf.float32),
            pattern + '_context1': tf.FixedLenFeature([4], tf.float32),
            pattern + '_context2': tf.FixedLenFeature([4], tf.float32),
            pattern + '_context3': tf.FixedLenFeature([4], tf.float32),
            pattern + '_context4': tf.FixedLenFeature([4], tf.float32),
            pattern + '_context5': tf.FixedLenFeature([4], tf.float32),
            pattern + '_context6': tf.FixedLenFeature([4], tf.float32),
            pattern + '_context7': tf.FixedLenFeature([4], tf.float32),
            pattern + '_semantic': tf.FixedLenFeature([32], tf.float32),
            'label': tf.FixedLenFeature([1], tf.float32)
        })

    #从原始图像数据解析出像素数组，根据图像尺寸还原图像放在反向传播过程中
    # print(features['label'])
    decoded_img = [features[pattern + '_img' + str(i)] for i in range(8)]
    #恢复其他数据
    context = [features[pattern + '_context' + str(i)] for i in range(8)]
    semantic = features[pattern + '_semantic']
    label = features['label']

    return decoded_img[0], decoded_img[1], decoded_img[2], decoded_img[
        3], decoded_img[4], decoded_img[5], decoded_img[6], decoded_img[
            7], context[0], context[1], context[2], context[3], context[
                4], context[5], context[6], context[7], semantic, label


#制作数据集
def make_dataset(file_path, pattern):

    dataset = tf.data.TFRecordDataset(file_path)
    dataset = dataset.map(lambda x: parser(x, pattern))
    if pattern == 'train':
        #如果是训练集，则打乱样例
        shuffer_buffer = 100
        dataset = dataset.shuffle(shuffer_buffer)
        #一次取 BATCH_SIZE 组数据，如果不够BATVCH_SIZE，则舍弃这组batch
        dataset = dataset.apply(
            tf.contrib.data.batch_and_drop_remainder(BATCH_SIZE))
        epoch = 10
        dataset = dataset.repeat(epoch)
    else:
        #如果是验证集或者是测试集，不打乱样例
        #每次验证时，从头取一个batch，保证每次取得的数据是一样的
        dataset = dataset.apply(
            tf.contrib.data.batch_and_drop_remainder(BATCH_SIZE))
    iterator = dataset.make_initializable_iterator()
    return iterator


def backward(stop, verifications, ckpt):
    # tf.reset_default_graph()
    #Local_CNN and LSTM paraments
    x = []
    xe = []
    for i in range(8):
        x.append(
            tf.placeholder(tf.float32, [
                BATCH_SIZE, DMVST_Net_CNN.IMAGE_SIZE, DMVST_Net_CNN.IMAGE_SIZE,
                DMVST_Net_CNN.NUM_CHANNELS
            ], name='x'+str(i)))
        xe.append(
            tf.placeholder(tf.float32, [None, DMVST_Net_forward.CONTEXT_NODE], name='xe'+str(i)))
    #Semantic paraments
    xm0 = tf.placeholder(tf.float32, [None, DMVST_Net_Semantic.INPUT_NODE],name='xm0')
    #label
    y_ = tf.placeholder(tf.float32, [None, DMVST_Net_forward.OUTPUT_NODE],name='y_')
    #后期完成语义视图的搭建后会修改 forward 函数，增加输入参数
    y = DMVST_Net_forward.forward(
        x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], xe[0], xe[1], xe[2],
        xe[3], xe[4], xe[5], xe[6], xe[7], xm0, True, BATCH_SIZE, REGULIZER)
    global_step = tf.Variable(0, trainable=False)


    #定义损失函数, 使用正则化缓解过拟合现象，把参数 w 的正则化加到总 loss 中
    smapl = tf.reduce_mean(tf.square(tf.divide(
        (y_ / 229 - y), y_ / 229))) * GARMA
    msl = tf.reduce_mean(tf.square(y_/229 - y))
    loss0 = smapl + msl
    loss = loss0 + tf.add_n(tf.get_collection('losses'))

    #设置指数衰减学习率，优化梯度下降步长
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        26558 / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    #梯度下降方法用 Adam，包括 l2 正则化损失
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(
        loss, global_step=global_step)

    #设置滑动平均，增加模型泛化性
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    #在来训练中添加update_ops以便在每一次训练完后及时更新BN的参数。
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
    with tf.control_dependencies([train_step, ema_op]):
        with tf.control_dependencies([train_step, ema_op]):
            train_op = tf.no_op(name='train')
    
    saver = tf.train.Saver(var_list=tf.global_variables())

    #evaluation metric
    rmse = tf.reduce_mean(tf.square(y * 229 - y_),name='rmse')

    #定义读取训练数据的数据集
    iterator = make_dataset(train_files, 'train')
    i0, i1, i2, i3, i4, i5, i6, i7, c0, c1, c2, c3, c4, c5, c6, c7, m, l = iterator.get_next(
    )
    img = [i0, i1, i2, i3, i4, i5, i6, i7]
    c = [c0, c1, c2, c3, c4, c5, c6, c7]

    #定义读取验证数据的数据集
    iterator1 = make_dataset(verify_files, 'verify')
    vi0, vi1, vi2, vi3, vi4, vi5, vi6, vi7, vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vm, vl = iterator1.get_next(
    )
    vimg = [vi0, vi1, vi2, vi3, vi4, vi5, vi6, vi7]
    vc = [vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7]

    print("数据集制作完毕，开始训练")

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        sess.run(tf.local_variables_initializer())
        sess.run(iterator.initializer)
        sess.run(iterator1.initializer)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        #开启线程协调器，多线程提高批获取的效率
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print('Training is start')

        def Run_sess(target, img, context, semantic, label):
            return sess.run(
                target,
                feed_dict={
                    x[0]: img[0],
                    x[1]: img[1],
                    x[2]: img[2],
                    x[3]: img[3],
                    x[4]: img[4],
                    x[5]: img[5],
                    x[6]: img[6],
                    x[7]: img[7],
                    xe[0]: context[0],
                    xe[1]: context[1],
                    xe[2]: context[2],
                    xe[3]: context[3],
                    xe[4]: context[4],
                    xe[5]: context[5],
                    xe[6]: context[6],
                    xe[7]: context[7],
                    xm0: semantic,
                    y_: label
                })

        for i in range(TRAIN_STEPS):
            try:
                img1,c1,m1,l1=sess.run([img,c,m,l])
                reshaped_i = []  #用于训练
                for k in range(8):
                    reshaped = np.reshape(
                        img1[k].eval(),
                        (BATCH_SIZE, DMVST_Net_CNN.IMAGE_SIZE,
                         DMVST_Net_CNN.IMAGE_SIZE, DMVST_Net_CNN.NUM_CHANNELS))
                    reshaped_i.append(reshaped)

                _, r, loss_value, step, learn = Run_sess(
                    [train_op, rmse, loss, global_step,learning_rate], reshaped_i, c1, m1, l1)
                r=math.sqrt(r)

                #每10轮把输出结果保存到一个日志文件
                if step % 10 == 0:
                    with codecs.open(record_file, 'a', 'utf-8') as writer:
                        writer.write(
                            "After " + str(step) +
                            " training step(s), loss on training batch is " +
                            str(loss_value) + ", rmse is " + str(r) + ".\n")

            #验证过程，当训练样本全部喂入神经网络一遍后保存一次模型并验证一次，每验证一次就要重启一次backword函数
            #每验证一次记录一次验证的RMSE，如果验证次数超过10次，查看倒数第11次记录的验证RMSE
            #是否小于近10次记录的RMSE，如果小于训练终止，认为倒数第11次保存的模型为最优模型
            except tf.errors.OutOfRangeError:
                print(
                    "\nAfter %d training step(s), loss on training batch is %g, rmse is %g."
                    % (step, loss_value, r))
                saver.save(
                    sess,
                    os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                    global_step=global_step)

                total_rmse_verify = 0.
                verify_step = 0
                for i in range(10000):
                    try:
                        vimg1,vc1,vm1,vl1=sess.run([vimg,vc,vm,vl])
                        reshaped_i1 = []  #用于验证
                        for k in range(8):
                            reshaped1 = np.reshape(
                                vimg1[k].eval(),
                                (BATCH_SIZE, DMVST_Net_CNN.IMAGE_SIZE,
                                DMVST_Net_CNN.IMAGE_SIZE, DMVST_Net_CNN.NUM_CHANNELS))
                            reshaped_i1.append(reshaped1)

                        pre, _1, loss_value1, p = Run_sess([y, y_, loss, rmse],
                                                        reshaped_i1, vc1, vm1, vl1)
                        total_rmse_verify += p
                        verify_step += 1
                    except tf.errors.OutOfRangeError:
                        p = total_rmse_verify / verify_step 
                        p = math.sqrt(p)

                        if os.path.exists(verify_file):
                            with codecs.open(verify_file, 'r',
                                            'utf-8') as read:
                                content = [s.strip() for s in read.readlines()]
                            verifications = int(
                                content[-2].split()[3].split('t')[0]) + 1
                            if verifications>10:
                                select_rmse = float(content[-9*3-2].split()[-1])
                                previous=[p]
                                for h in range(0, 9):
                                    previous.append(float(content[-h*3-2].split()[-1]))
                                min0 = min(previous)
                                if select_rmse < min0:
                                    print("\n\nThe number of the best model: " + content[-9*3-2].split()[3].split('t')[0] + "\n\n")
                                    stop=1
                        else:
                            verifications += 1

                        print(
                            "This is the %dth verification, loss on this verifying dataset's last batch is %g, average rmse is %g"
                            % (verifications, loss_value1, p))
                        print("the prediction value is: ", pre[-1] * 229, _1[-1],
                            pre[2] * 229, _1[2])
                        
                        with codecs.open(verify_file, 'a', 'utf-8') as w:
                            w.write("After " + str(step) +
                                    " training step(s), loss on training batch is " +
                                    str(loss_value) + ", rmse is " + str(r)+', learning_rate is ' + str(learn) + "\n")
                            w.write("This is the " + str(verifications) +
                                    "th verification, loss on this verifying dataset's last batch is "
                                    + str(loss_value1) + ", average rmse is " + str(p) + "\n")
                            w.write('stop: ' + str(stop) + '\n')
                        break
                break

        #关闭线程协调器
        coord.request_stop()
        coord.join(threads)


def main():
    stop = 0  #是否退出训练的标记，为1保存模型退出训练反向传播过程结束
    verifications = 0  #记录验证次数
    if stop == 1:
        print(
            'The model is very close to the best state, it has great probability to be overfit if we continue traing, so we will stop training!'
        )
    else:
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        backward(stop, verifications, ckpt)


if __name__ == '__main__':
    main()
