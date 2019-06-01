#训练样例数据是从1月1日第8个时刻开始到1月21日最后一个时刻，
#所以决定生成多个TFRecord文件，总共(20*48+41)*265个样例，所以生成265个TFRecord文件，
#每个TFRecord文件对应一个地点
#每个TFRecord文件里头有20*48+41=1001个样例，每个样例对应一个时刻，
#每个样例有18个属性：
'''
'train_img0-7':一个9*9*1长的二进制向量对应的值
'train_context0-7':一个4维的浮点向量对应的值
'train_semantic':一个32维向量，即权重图经LINE处理后得到的低维语义向量对应的值
'label':标签数据，是一个浮点型数据
'''

#验证样例数据是从1月22日第一个时刻到1月25最后一个时刻
#验证样例数据的处理方式和训练样例数据的处理方式一样，只是TFRecord文件和样例属性的命名方式不同
#每个测试用的TFRecord文件用 4*48 个样例

#测试样例数据是从1月26日第一个时刻到1月28日最后一个时刻，因为最后3天没有对应的语义向量，所以最后3天的数据舍弃
#测试样例数据的处理方式和训练样例数据的处理方式一样，只是TFRecord文件和样例属性的命名方式不同
#每个测试用的TFRecord文件用 3*48 个样例

import tensorflow as tf
import numpy as np
from PIL import Image
import codecs
import os
import DMVST_DATA
import DMVST_DATA_test
import DMVST_DATA_verify

LOCATION_LABEL = '../img_value.vocab'
CONTEXT_FILE = '../CONTEXT_DATA.vocab'
SEMANTIC_DATA_PATH = '../Embed/'
SAVED_DATA_PATH = './data-fortest/'
IMAGE_PIXEL_PATH = '../CNN_IMG_PIXEL_DATA/'

#恢复标签数据，一个地点在一个时刻标签数据为一个地点在下一个时刻的乘车事件的数量
with codecs.open(LOCATION_LABEL, 'r', 'utf-8') as labels:
    labelValue = [w.strip() for w in labels.readlines()]

label_temp = []
total_label = []
for i in range(len(labelValue)):
    label_split = [float(w) for w in labelValue[i].split()]
    for n in range(len(label_split)):
        if label_split[n] == 0.:
            label_split[n] = 1.
    total_label.append(label_split)
    label_temp.append(label_split[1:])
    label_split = []
for i in range(265):
    for j in range(31):
        if j < 30:
            label_temp[j * 265 + i].append(total_label[(j+1) * 265 + i][0])
        else:
            #每个地点在 1 月最后一个时刻的标签为 2 月第一个时刻的数据，
            #这里用1月2号第一个时刻的数据代替
            label_temp[j * 265 + i].append(total_label[i + 265][0])

label_Value = np.array(label_temp)
label_Value = label_Value.reshape(31, 265, 48)
print('label datas have been restored')

#恢复上下文（环境）数据
with codecs.open(CONTEXT_FILE, 'r', 'utf-8') as context:
    context = [w.strip() for w in context.readlines()]

context_value = []
#266*31行，因为重写代码过于麻烦，所以用的旧代码，但不影响结果
for i in range(len(context)):
    context_split = [w for w in context[i].split()]
    context_temp = []
    for j in range(len(context_split)):
        context_temp0 = [float(a) for a in context_split[j].split('-')]
        context_temp.append(context_temp0)
        context_temp0 = []
    context_value.append(context_temp)

context_value = np.array(context_value)
context_value = context_value.reshape(31, 266, 48, 4)
print('context datas have been restored')

#创建保存训练数据和测试数据的文件夹
train_data_path = SAVED_DATA_PATH + 'train_data/'
verify_data_path = SAVED_DATA_PATH + 'verify_data/'
test_data_path = SAVED_DATA_PATH + 'test_data/'
isExists1 = os.path.exists(train_data_path)
isExists2 = os.path.exists(verify_data_path)
isExists3 = os.path.exists(test_data_path)
if not isExists1:
    #如果路径不存在则新建这个路径
    os.makedirs(train_data_path)
if not isExists2:
    os.makedirs(verify_data_path)
if not isExists3:
    os.makedirs(test_data_path)


def nums_func(pattern):
    num_shards = 265  #总共生成文件数
    if pattern == 'train':
        instances_per_shared = 20 * 48 + 41  #每个训练文件中有多少个样例
        model = DMVST_DATA
        data_path = train_data_path
    elif pattern == 'verify':
        instances_per_shared = 4 * 48  #每个验证文件中有多少个样例
        model = DMVST_DATA_verify
        data_path = verify_data_path
    else:
        instances_per_shared = 3 * 48  #每个测试文件有多少个样例
        model = DMVST_DATA_test
        data_path = test_data_path
    return num_shards, instances_per_shared, model, data_path


#生成二进制型属性
def _bytes_features(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


#生成浮点型属性-label
def _float_features(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


#生成浮点型数组属性-img0-7, context0-7, semantic
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


#生成TFRecord文件
def make_tfrecords(pattern):
    num_shards, instances_per_shared, model, data_path = nums_func(pattern)

    #得到所有语义数据
    semantic_value = []
    for i in range(num_shards):
        semantic_value.append(
            model.get_semantic_data(SEMANTIC_DATA_PATH, i+1))
    print('semantic datas for' + pattern + ' are ready',
          len(semantic_value[0]))

    #得到所有上下文数据
    context_value0 = []
    for i in range(num_shards):
        context_datas = model.get_context_data(context_value, i+1)
        #print(len(context_datas),len(context_datas[1]))
        context_data = []
        for j in range(instances_per_shared):
            single_loc = []
            for k in range(8):
                single_loc.append(context_datas[j][k])
            context_data.append(single_loc)
        context_value0.append(context_datas)
    print('context datas for ' + pattern + ' are ready', len(context_value0),
          len(context_value0[1]), len(context_value0[0][1]),
          len(context_value0[1][1][1]))

    #得到所有的图片数据，以及标签数据
    img_names = []
    labels = []
    for i in range(num_shards):
        img_name, label, _ = model.get_localCNN_files(IMAGE_PIXEL_PATH, label_Value, i+1)
        img_names.append(img_name)
        labels.append(label)
        print("loc%d's img_names and labels for " % (i + 1) + pattern +
              " are ready")

    img_raw = []
    imgs_size = []
    for i in range(num_shards):
        img_contents = []
        size1 = []
        for j in range(instances_per_shared):
            img_content = []
            size2 = []
            for k in range(8):
                # #此方法是读img的数据然后转为二进制
                # #打开图片
                # img = Image.open(img_names[i][j][k])
                # # 将每张图片转换为二进制数据
                # img_content.append(img.tobytes())
                # size2=img.size

                #此方法简单粗暴，直接读取存有像素值的文件
                with codecs.open(img_names[i][j][k], 'r', 'utf-8') as read:
                    img_data = [float(w) for w in read.readline().split()]
                    img_data_temp = []
                    for m in img_data:
                        img_data_temp.append(m)

                # #此方法是读取图片的像素矩阵，然后拉成81维向量
                # with tf.Session() as sess:
                #     img = tf.gfile.FastGFile(img_names[i][j][k], 'r').read()
                #     img_tensor=tf.image.decode_bmp(img)
                #     img_data=img_tensor.eval()
                #     img_data=np.array(img_data)
                #     img_data=np.reshape(img_data,(9*9*1))
                #     img_data_temp=[]
                #     for pic in range(len(img_data)):
                #         img_data_temp.append(img_data[pic])
                #     train_img_content.append(img_data_temp)
                img_content.append(img_data_temp)
            print("loc%d's %dth img data for " % (i + 1, j + 1) + pattern +
                  " is done")
            img_contents.append(img_content)
            size1.append(size2)
        img_raw.append(img_contents)
        imgs_size.append(size1)
    print('img datas and labels ' + pattern + ' are ready ')
    # train_img_raw=train_img_names

    #生成TFRecords
    for i in range(num_shards):
        filename = (data_path + pattern + '.tfrecords-loc%.3d' % (i + 1))
        writer = tf.python_io.TFRecordWriter(filename)
        for j in range(instances_per_shared):
            #过滤掉需求值小于10的样本
            # if labels[i][j]<10.:
            #     continue
            #将数据封装成 Example 结构并写入 TFRecord 文件
            imgData = []
            contextData = []
            for k in range(8):
                imgData.append(_float_feature(img_raw[i][j][k]))
                contextTemp = []
                for m in range(4):
                    contextTemp.append(context_value0[i][j][k][m])
                contextData.append(_float_feature(contextTemp))

            semanticTemp = []
            for m in range(32):
                semanticTemp.append(semantic_value[i][j][m])
            semanticData = _float_feature(semanticTemp)

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        pattern + '_img0': imgData[0],
                        pattern + '_img1': imgData[1],
                        pattern + '_img2': imgData[2],
                        pattern + '_img3': imgData[3],
                        pattern + '_img4': imgData[4],
                        pattern + '_img5': imgData[5],
                        pattern + '_img6': imgData[6],
                        pattern + '_img7': imgData[7],
                        pattern + '_context0': contextData[0],
                        pattern + '_context1': contextData[1],
                        pattern + '_context2': contextData[2],
                        pattern + '_context3': contextData[3],
                        pattern + '_context4': contextData[4],
                        pattern + '_context5': contextData[5],
                        pattern + '_context6': contextData[6],
                        pattern + '_context7': contextData[7],
                        pattern + '_semantic': semanticData,
                        'label': _float_features(labels[i][j]),
                        # pattern + '_width':tf.train.Feature(int64_list=tf.train.Int64List(value=[imgs_size[i][j][0]])),
                        # pattern + '_height':tf.train.Feature(int64_list=tf.train.Int64List(value=[imgs_size[i][j][1]])),
                        # pattern + 'channels':tf.train.Feature(int64_list=tf.train.Int64List(value=[1]))
                    }))
            writer.write(example.SerializeToString())
        print(filename + ' has been generated')
        writer.close()


make_tfrecords('train')
# make_tfrecords('verify')
# make_tfrecords('test')