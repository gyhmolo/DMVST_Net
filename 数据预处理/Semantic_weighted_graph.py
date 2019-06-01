#得到各个地区的权重图
import numpy as np
import codecs
import math
import os

DEMAND_DATA = '../demand_pattern.vocab'
SEMANTIC_INPUT_FILE = '../SEMANTIC_GRAPH_INPUT_FILE/'


#实现 DTW 算法
def dtw(s1, s2):
    r, c = len(s1), len(s2)
    D1 = np.zeros((r, c))

    for i in range(r):
        for j in range(c):
            temp = s1[i] - s2[j]
            if temp > 0:
                D1[i][j] = temp
            else:
                D1[i][j] = 0 - temp

    for i in range(r):
        for j in range(c):
            if i - 1 >= 0 and j - 1 >= 0:
                D1[i][j] += min(D1[i - 1][j - 1], D1[i][j - 1], D1[i - 1][j])
            elif i - 1 >= 0 and j - 1 < 0:
                D1[i][j] += D1[i - 1][j]
            elif i - 1 < 0 and j - 1 >= 0:
                D1[i][j] += D1[i][j - 1]
            else:
                D1[i][j] = D1[i][j]

    return D1[-1][-1]


#如果文件路径不存在则创建路径
isExists = os.path.exists(SEMANTIC_INPUT_FILE)
if not isExists:
    os.makedirs(SEMANTIC_INPUT_FILE)

#首先恢复 demand_pattern 数据
with codecs.open(DEMAND_DATA, 'r', 'utf-8') as dd:
    demandValue0 = [w.strip() for w in dd.readlines()]

demandValue = []
for i in range(len(demandValue0)):
    demandValue_temp = [float(w) for w in demandValue0[i].split()]
    demandValue.append(demandValue_temp)
    demandValue_temp = []
demandValue = np.array(demandValue)

#每个地点每一周都有对应的权重图文件，共266*4个文件
#每个文件的命名格式：地区id_周数
#每个地区在每周有265个权重值，舍弃不合适的权重组合，保存到文件中
#每行为：预测的地区(本地区) 其他地区(包括本地区) 权重值
for i in range(265):
    for j in range(4):
        filename = SEMANTIC_INPUT_FILE + str(i + 1) + '_' + str(j + 1)
        with codecs.open(filename, 'w', 'utf-8') as file_output:
            dtwValue = []
            num = 0  #记录写入了多少个组合
            dtwtemp = []  #记录写入的权重信息对应的DTW，以找出最小的DTW
            for k in range(265):
                dtwValue = (dtw(demandValue[i * 4 + j],
                                demandValue[k * 4 + j]))
                if dtwValue == 0:
                    dtwtemp.append(100000000)
                else:
                    dtwtemp.append(dtwValue)
                #如果两个地区在同一时间上的DTW大于100，认为相似度过低舍弃数据
                #由于是无向图，所以每次写入两行
                if dtwValue <= 100. and dtwValue > 0:
                    weight = math.exp(-dtwValue)
                    num += 1
                    file_output.write(
                        str(i + 1) + ' ' + str(k + 1) + ' ' + str(weight) +
                        '\n')
                    file_output.write(
                        str(k + 1) + ' ' + str(i + 1) + ' ' + str(weight) +
                        '\n')
            #有些地区组合在同一时间段的DTW可能全部大于100，如果发生这种情况，文件为空
            #这时往文件里写入DTW最小的权重信息
            if num == 0:
                dtwtemp = np.array(dtwtemp)
                lock = np.argmin(dtwtemp)
                minvalue = dtwtemp.min()
                minweight = math.exp(-minvalue)
                file_output.write(
                    str(lock + 1) + ' ' + str(i + 1) + ' ' + str(minweight) +
                    '\n')
                file_output.write(
                    str(i + 1) + ' ' + str(lock + 1) + ' ' + str(minweight) +
                    '\n')
