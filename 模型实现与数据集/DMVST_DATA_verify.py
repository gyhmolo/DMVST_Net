
import os
import codecs
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def get_localCNN_files(file_dir, label_total, local_id):
    #从2018年1月22日第1个时刻提取图片
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
            if int(name[1]) >= 20 and int(name[1]) < 25:
                event_graph.append(file_dir + file)
                time = int(name[2].split('.')[0])
                event_label.append(label_total[int(name[1])][int(
                    name[0])-1][time])
            else:
                continue

    #每个地点有4*48个样本参与验证
    images = []
    labels = []
    for i in range(48, len(event_graph)):
        if i < len(event_graph) - 1:
            temp = [w for w in event_graph[i - 7:i + 1]]
            images.append(temp)
            labels.append(event_label[i])
        else:
            temp = [w for w in event_graph[i - 7:]]
            images.append(temp)
            labels.append(event_label[i])

    return images, labels, len(images)


def get_context_data(context_value, local_id):
    context_data_all = []
    for day in range(20, 25):
        for interval in range(0, 48):
            context_data_all.append(
                context_value[day][local_id][interval])
                
    # for loc_id in range(266):
    #     if loc_id < local_id:
    #         continue
    #     elif loc_id > local_id:
    #         break
    #     else:
    #         for day in range(20, 25):
    #             for interval in range(0, 48):
    #                 context_data_all.append(
    #                     context_value[loc_id][day][interval])

    context_data = []
    for i in range(48, len(context_data_all)):
        if i < len(context_data_all) - 1:
            temp = [c for c in context_data_all[i - 7:i + 1]]
            context_data.append(temp)
        else:
            temp = [c for c in context_data_all[i - 7:]]
            context_data.append(temp)

    return context_data


def get_semantic_data(file_path, local_id):
    #取指定地点的第四个文件
    for file in os.listdir(file_path):
        name = file.split('_')
        # print(file)
        if int(name[0]) < local_id:
            continue
        elif int(name[0]) > local_id:
            continue
        else:
            if int(name[1]) == 4:
                semantic_file = file_path + file

    #每个文件有不同的行数，每行有33个元素，每行第一个为与指定地点相比较的地点
    #每次喂入神经网络的只是单个的32维向量
    #修订后的程序不再分析地点0的数据
    #所以这里把每个文件的向量相加求平均得到一个新的32维向量喂入神经网络
    # print(semantic_files)
    semantic_data = []
    with codecs.open(semantic_file, 'r', 'utf-8') as read:
        lines = [line.strip() for line in read.readlines()]
    matrix = [[0.] * 32] * len(lines)
    matrix = np.array(matrix)
    for i in range(len(lines)):
        temp = [float(w) for w in lines[i].split()]
        temp = temp[1:]
        matrix[i-1] = temp
    sum_vector = matrix.sum(axis=0)
    mean_vector = sum_vector / len(lines)

    for i in range(4 * 48):
        semantic_data.append(mean_vector)

    return np.array(semantic_data)
