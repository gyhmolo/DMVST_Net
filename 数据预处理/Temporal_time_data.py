#获取节假日信息和星期几数据
import numpy as np
import codecs

TIME_DATA = '../time.vocab'

#标注2018年1月份的每一天是星期几
week = [
    1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4,
    5, 6, 7, 1, 2, 3
]

#标注2018年1月份美国的节假日(不包括双休)
holidays = [
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0
]

#生成时间信息
time_info = []
for i in range(31):
    temp = str(week[i]) + '-' + str(holidays[i])
    time_info.append(temp)

#把时间信息扩展为 31*266*48 的矩阵
counts_time = [[["0-0"] * 48] * 266] * 31
counts_time = np.array(counts_time)
for i in range(31):
    for j in range(266):
        temp = [time_info[i]] * 48
        temp = np.array(temp)
        counts_time[i][j] = temp

#将处理好的时间信息数据特征保存到 time.vocab 文件
counts_time_str = []
data = ""
for i in range(31):
    for j in range(266):
        for k in range(48):
            if k < 47:
                data = data + counts_time[i][j][k] + ' '
            else:
                data = data + counts_time[i][j][k]
        counts_time_str.append(data)
        data = ""

with codecs.open(TIME_DATA, 'w', 'utf-8') as file_output:
    for data in counts_time_str:
        file_output.write(data + '\n')
