#该文件主要用于提取数据供神经网络应用
import os
import codecs
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def get_localCNN_files(file_dir, label_total, local_id):
    #从2018年1月1日第8个时刻提取图片
    #取前21天的数据作为训练数据

    #定义存放图片名的列表
    event_graph = []
    #定义存放图片对应标签的列表
    event_label = []

    #提取所需地区的文件和标签
    for file in os.listdir(file_dir):
        name = file.split('_')
        if int(name[0]) != local_id:
            continue
        else:
            if int(name[1]) < 21:
                event_graph.append(file_dir + file)
                time = int(name[2].split('.')[0])
                event_label.append(label_total[int(name[1])][int(name[0])-1][time])
            else:
                continue

    #每个地点的前7个时刻不参与预测
    #每个地点有21*48-7个样本参与训练
    images = []
    labels = []
    for i in range(7, len(event_graph)):
        if i < len(event_graph) - 1:
            temp = [w for w in event_graph[i - 7:i + 1]]
            images.append(temp)
            labels.append(event_label[i])
        else:
            temp = [w for w in event_graph[i - 7:]]
            images.append(temp)
            labels.append(event_label[i])
        temp = []

    return images, labels, len(event_graph) - 7


def get_context_data(context_value, local_id):
    context_data_all = []
    for day in range(21):
        for interval in range(0, 48):
            # print(loc_id,day,interval)
            context_data_all.append(
                context_value[day][local_id][interval])

    # for loc_id in range(266):
    #     if loc_id < local_id:
    #         continue
    #     elif loc_id > local_id:
    #         break
    #     else:
    #         for day in range(0, 21):
    #             for interval in range(0, 48):
    #                 # print(loc_id,day,interval)
    #                 context_data_all.append(
    #                     context_value[loc_id][day][interval])
    #             if day == 20:
    #                 break
    # print('kkk', len(context_data_all),len(context_data_all[1]))

    context_data = []
    for i in range(7, len(context_data_all)):
        if i < len(context_data_all) - 1:
            temp = [c for c in context_data_all[i - 7:i + 1]]
            context_data.append(temp)
        else:
            temp = [c for c in context_data_all[i - 7:]]
            context_data.append(temp)
        temp = []

    # print('kkk', len(context_data))

    return context_data


def get_semantic_data(file_path, local_id):
    #取指定地点的前三个文件
    semantic_files = []
    for file in os.listdir(file_path):
        name = file.split('_')
        # print(file)
        if int(name[0]) < local_id:
            continue
        elif int(name[0]) > local_id:
            continue
        else:
            if int(name[1]) < 4:
                semantic_files.append(file_path + file)
            else:
                continue
    #每个文件有不同的行数，每行有33个元素，每行第一个为与指定地点相比较的地点
    #每次喂入神经网络的只是单个的32维向量
    #所以这里把每个文件的向量相加求平均得到一个新的32维向量喂入神经网络
    # print(semantic_files)
    semantic_data = []
    for i in range(3):
        with codecs.open(semantic_files[i], 'r', 'utf-8') as read:
            lines = [line.strip() for line in read.readlines()]
        matrix = [[0.] * 32] * len(lines)
        matrix = np.array(matrix)
        for j in range(len(lines)):
            temp = [float(w) for w in lines[j].split()]
            temp = temp[1:]
            matrix[j-1] = temp
        sum_vector = matrix.sum(axis=0)
        mean_vector = sum_vector / len(lines)

        for k in range(7 * 48):
            semantic_data.append(mean_vector)

    semantic_train = []
    for i in range(7, len(semantic_data)):
        semantic_train.append(semantic_data[i])

    return np.array(semantic_train)


#定义我们自己的 next_batch函数
class DataSet():
    def __init__(self, images, context, semantic, labels, num_examples):
        self._images = images
        self._labels = labels
        self._context = context
        self._semantic = semantic
        self._epochs_completed = 0  # 完成遍历轮数
        self._index_in_epochs = 0  # 调用next_batch()函数后记住上一次位置
        self._num_examples = num_examples  # 训练样本数

    def next_batch(self, batch_size, fake_data=False, shuffle=False):
        start = self._index_in_epochs
        if self._epochs_completed == 0 and start == 0 and not shuffle:
            index0 = np.arange(self._num_examples)
            # np.random.shuffle(index0)
            self._images = np.array(self._images)
            # self._context = np.array(self._context)
            # self._semantic = np.array(self._semantic)
            self._labels = np.array(self._labels)

        # print(self._images.shape, self._context.shape, self._semantic.shape,
        #       self._labels.shape)

        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            # images_rest_part = self._images[start:self._num_examples]
            # context_rest_part = self._context[start:self._num_examples]
            semantic_rest_part = self._semantic[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]

            start = 0
            new_start = self._num_examples - batch_size
            new_end = new_start + batch_size - rest_num_examples
            # images_new_part = self._images[new_start:new_end]
            # context_new_part = self._context[new_start:new_end]
            semantic_new_part = self._semantic[new_start:new_end]
            labels_new_part = self._labels[new_start:new_end]

            images_batch = []
            context_batch = []
            semantic_batch = []
            labels_batch = []
            for i in range(8):
                images_rest_part = self._images[i][start:self._num_examples]
                images_new_part = self._images[i][new_start:new_end]
                images_batch.append(
                    np.concatenate((images_new_part, images_rest_part),
                                   axis=0))
                context_rest_part = self._context[i][start:self._num_examples]
                context_new_part = self._context[i][new_start:new_end]
                context_batch.append(
                    np.concatenate((context_new_part, context_rest_part),
                                   axis=0))
            semantic_batch = np.concatenate(semantic_new_part,
                                            semantic_rest_part)
            labels_batch = np.concatenate((labels_new_part, labels_rest_part),
                                          axis=0)

            return images_bach, context_batch, semantic_batch, labels_batch

        else:
            self._index_in_epochs += batch_size
            end = self._index_in_epochs
            images_batch = []
            context_batch = []
            semantic_batch = []
            for i in range(8):
                images_batch.append(self._images[i][start:end])
                print(i)
                print(self._images[i])
                print(self._images.shape, self._context.shape)
            print(images_batch[1].shape)
            context_batch = self._context[start:end]
            semantic_batch = self._semantic[start:end]
            labels_batch = self._labels[start:end]
            return images_batch, context_batch, semantic_batch, labels_batch


def get_batch(counts, image, label, context, semantic, image_W, image_H,
              batch_size, capacity):

    #类型变换，暂且不知道有什么用，先变了再说
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # context = np.array(context)
    # semantic = np.asarray(semantic)

    #加入队列，这与tensorflow处理数据的机制有关
    #详情见https://blog.csdn.net/dcrmg/article/details/79776876
    input_queue = tf.train.slice_input_producer(
        [image, label, context, semantic])

    print(input_queue[2].shape)

    label = input_queue[1]
    context = input_queue[2]
    semantic = input_queue[3]
    #解码图片
    image_contents = []
    image0 = []
    image1 = []
    image = []
    for i in range(8):
        image_contents.append(tf.read_file(input_queue[0][i]))
        image0.append(tf.image.decode_bmp(image_contents[i], channels=1))
        image1.append(
            tf.image.resize_image_with_crop_or_pad(image0[i], image_W,
                                                   image_H))
        image.append(tf.image.per_image_standardization(image1[i]))
    #resize，对之后的测试有用
    # image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    #对解码的图片进行标准化处理
    #图像标准化是将数据通过去均值实现中心化的处理，
    #根据凸优化理论与数据概率分布相关知识，数据中心化符合数据分布规律，
    #更容易取得训练之后的泛化效果, 数据标准化是数据预处理的常见方法之一
    # image = tf.image.per_image_standardization(image)

    dataset = DataSet(image, label, context, semantic, counts)
    image_batch, context_batch, semanric_batch, label_batch = dataset.next_batch(
        batch_size)

    # image_batch, label_batch = tf.train.batch([image, label],
    #                                           batch_size=batch_size,
    #                                           num_threads=1,
    #                                           capacity=capacity)

    # label_batch = tf.reshape(label, [batch_size])
    #获得两个batch，两个batch即为传入神经网络的数据
    return image_batch, context_batch, semanric_batch, label_batch


def get_img_Value(image,image_W,image_H):
    image_value_total = []
    for interval in range(len(image)):
        image_value = []
        for i in range(8):
            image_contents = tf.read_file(image[interval][i])
            image0 = tf.image.decode_bmp(image_contents,channels=1)
            image1 = tf.image.resize_image_with_crop_or_pad(
                image0, image_W, image_H)
            image_value.append(tf.image.per_image_standardization(image1))
        image_value_total.append(image_value)
        print("%dth group graph data is done" %(interval +1))
    # image_value_total = np.array(image_value_total)
    return image_value_total



