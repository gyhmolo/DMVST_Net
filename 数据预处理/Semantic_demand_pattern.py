#构建每个地点的 demand pattern
import numpy as np
import codecs

PICK_DATA = '../img_value.vocab'
DEMAND_VALUE = '../demand_pattern.vocab'

#取 1 月份的前四周，每个地点对应4个7维向量，每个向量的元素为每天的平均事件数
#最后三天取出备用
with codecs.open(PICK_DATA, 'r', 'utf-8') as dp:
    pickdata = [p.strip() for p in dp.readlines()]

#求每个地点每天的事件数平均值

#先恢复数据
int_pick = []
for i in range(len(pickdata)):
    int_pick_temp = [int(w) for w in pickdata[i].split()]
    int_pick.append(int_pick_temp)
    int_pick_temp = []

average0 = []
for i in range(len(int_pick)):
    single_average = np.mean(int_pick[i])
    average0.append(single_average)

average = []
for i in range(265):
    for j in range(4):
        temp = []
        for k in range(7):
            pointer = i * 31 + j * 7 + k
            temp.append(average0[pointer])
        average.append(temp)

# average 为266*4行7列的矩阵，每4行对应一个地点的4个demand_pattern
average = np.array(average)

# 将 demand_pattern 数据写入 demand_pattern.vocab 文件
average_str = []
data = ''
for i in range(4 * 265):
    for j in range(7):
        data = data + str(average[i][j]) + ' '
    average_str.append(data)
    data = ''

with codecs.open(DEMAND_VALUE, 'w', 'utf-8') as file_output:
    for data in average_str:
        file_output.write(data + '\n')
