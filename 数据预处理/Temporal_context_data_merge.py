#合并上下文数据
import numpy as np 
import codecs

WC_DATA = '../wc.vocab'
AQ_DATA = '../aq_value.vocab'
TIME_DATA = '../time.vocab'
CONTEXT_DATA = '../CONTEXT_DATA.vocab'

#恢复数据(31*266行，48列的矩阵)
#读出环境状况数据
with codecs.open(WC_DATA, 'r', 'utf-8') as wc_file:
    wcdata0 = [wc.strip() for wc in wc_file.readlines()]
wcdata = []
for i in range(len(wcdata0)):
    temp = wcdata0[i].split()
    wcdata.append(temp)
wcdata = np.array(wcdata)

#读出空气质量数据
with codecs.open(AQ_DATA, 'r', 'utf-8') as aq_file:
    aqdata0 = [aq.strip() for aq in aq_file.readlines()]
aqdata = []
for i in range(len(aqdata0)):
    temp = aqdata0[i].split()
    aqdata.append(temp)
# aqdata = np.array(aqdata)

#读出时间信息
with codecs.open(TIME_DATA, 'r', 'utf-8') as time_file:
    timedata0 = [time.strip() for time in time_file.readlines()]
timedata = []
for i in range(len(timedata0)):
    temp = timedata0[i].split()
    timedata.append(temp)
timedata = np.array(timedata)

#制作表示每个地点每天每个时间段的上下文数据三维矩阵
#第三维的每个数据就是每个地点每天每个时间段的上下文数据，格式是
#天气状况-空气质量-星期几-是否为节假日
#其中空气质量数据是被归一化的
context_counts = [[['00-%.5f-0-0' %0.12345] * 48] * 266] * 31
context_counts = np.array(context_counts)
for i in range(31):
    for j in range(266):
        for k in range(48):
            context_counts[i][j][k] = str(wcdata[i * 266 + j][k]) + '-' + ('%.5f'%float(aqdata[i * 266 + j][k])) + '-' + timedata[i * 266 + j][k]

#将处理好的上下文数据保存到 CONTEXT_DATA.vocab
context_counts_str = []
data=""
for i in range(31):
    for j in range(266):
        for k in range(48):
            if k < 47:
                data = data + context_counts[i][j][k] + ' '
            else:
                data = data + context_counts[i][j][k]
        context_counts_str.append(data)
        data = ""

with codecs.open(CONTEXT_DATA, 'w', 'utf-8') as file_output:
    for data in context_counts_str:
        file_output.write(data + '\n')