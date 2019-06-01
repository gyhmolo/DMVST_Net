#提取空气质量数据
import pandas as pd
import numpy as np
import codecs

AQ_VALUE = '../aq_value.vocab'

aq = pd.read_csv('../daily_aqi_by_county_2016.csv')
loc_id = pd.read_csv('../taxi+_zone_lookup.csv')

aq_colnames = aq.columns.tolist()
aq_temp = aq[[aq_colnames[0], aq_colnames[1], aq_colnames[4], aq_colnames[5]]]

state_str = aq_temp[aq_colnames[0]].tolist()
loc_str = aq_temp[aq_colnames[1]].tolist()
date_str = aq_temp[aq_colnames[4]].tolist()
aq_str = aq_temp[aq_colnames[5]].tolist()

aq_ny_2018_1 = []

for i in range(len(state_str)):
    temp_data = date_str[i].split('-')
    if state_str[i] == 'New York' and int(temp_data[1]) == 1:
        aq_ny_2018_1.append(
            [loc_str[i], int(temp_data[2]) - 1,
             int(aq_str[i])])

#由于出租车数据集中在曼哈顿、皇后区、布朗克斯和布鲁克林，这些都在市中心，所以
#对部分地区抽样以减少数据,用局部代替整体
#我选取布朗克斯和皇后区的数据，并把他们的空气质量指标求平均并成一个新的list，
#用该 list代替纽约市整体的环境数据

aq_ny_counts = [0.] * 31
aq_ny_counts = np.array(aq_ny_counts)
for i in range(31):
    aqValue = 0.
    timer = 0
    for j in range(len(aq_ny_2018_1)):
        if aq_ny_2018_1[j][1] == i and (aq_ny_2018_1[j][0] == 'Bronx'
                                        or aq_ny_2018_1[j][0] == "Queens"):
            timer += 1
            aqValue += float(aq_ny_2018_1[j][2])
    aqmean = aqValue // timer
    aq_ny_counts[i] = aqmean
aq_max = float(np.max(aq_ny_counts))
aq_ny_counts = aq_ny_counts/aq_max

#把空气质量数据扩展成为 31*266*48 的矩阵
counts_aq = [[[0.] * 48] *266] * 31
counts_aq = np.array(counts_aq)
for i in range(31):
    for j in range(266):
        temp=[aq_ny_counts[i]]*48
        counts_aq[i][j]=temp

#将处理好的空气质量数据特征保存到 aq_value.vocab 文件
counts_aq_str = []
data = ""
for i in range(31):
    for j in range(266):
        for k in range(48):
            if k < 47:
                data = data + str(counts_aq[i][j][k]) + ' '
            else:
                data = data + str(counts_aq[i][j][k])
        counts_aq_str.append(data)
        data = ""

with codecs.open(AQ_VALUE, 'w', 'utf-8') as file_output:
    for data in counts_aq_str:
        file_output.write(data + '\n')
