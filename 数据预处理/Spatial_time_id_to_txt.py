# 生成灰度图对应的像素值
import numpy as np
import codecs
import os

TOTAL_VALUE = '../total_value.vocab'
IMG_VALUE = '../img_value.vocab'
PIXEL_PATH = '../CNN_IMG_PIXEL_DATA/'

prednum = 265

# 如果图片路径不存在则创建路径
isExists = os.path.exists(PIXEL_PATH)
if not isExists:
    os.makedirs(PIXEL_PATH)

# 读出乘车数量（即空间特征）
with codecs.open(TOTAL_VALUE, 'r', 'utf-8') as f_img:
    imgValue = [w.strip() for w in f_img.readlines()]

# 恢复 counts
counts2 = []
maxs = []
maxs_index= []
for i in range(len(imgValue)):
    counts1 = [int(w) for w in imgValue[i].split()]
    counts2.append(counts1)
    maxs.append(max(counts1))
    maxs_index.append(np.argmax(np.array(counts1)))
    counts1 = []
max_value = max(maxs)
index=np.argmax(np.array(maxs))
maxvalue_index = [index, maxs_index[index]]
counts = []
for i in range(31):
    counts.append(counts2[i * len(counts2) // 31:(i + 1) * len(counts2) // 31])
counts = np.array(counts)

# 得到灰度数据counts_gray
normalize = [[[1. / max_value] * 48] * len(counts[0])] * 31
normalize = np.array(normalize)
counts_gray = counts * normalize
print(counts_gray.shape)
# counts_gray = np.flatten(counts * normalize)

# 生成 image_value.vocab
imgvalue_str = []
data = ""
for i in range(31):
    for j in range(prednum):
        for k in range(48):
            if k < 47:
                data = data + str(counts[i][(4+j//16) * 24+j % 16+4][k]) + ' '
            else:
                data = data + str(counts[i][(4+j//16) * 24+j % 16+4][k])
        imgvalue_str.append(data)
        data = ""

with codecs.open(IMG_VALUE, 'w', 'utf-8') as file_output:
    for i in range(len(imgvalue_str)):
        file_output.write(imgvalue_str[i] + '\n')

# 建立2018年1月每个地点每一天的每个时段的流量图，有265*31*48张图片，每张图片的命名规则：
# 地区id号(1-265)_天数(00-30)_时段编号(00-47)
# size = len(counts_gray[0])
# imgvalue_file = []
# pics = []
# for day in range(31):
#     for time_interval in range(48):
#         pic = []
#         for loc_id in range(prednum):
#             pic.append(counts_gray[day][(4+loc_id//16)
#                                         * 24+loc_id % 16+4][time_interval])
#         imgvalue_file.append(pic)
#         for loc_id in range(len(pic)):
#             pic_value = np.array([[0.]*9]*9)
#             # 填充灰度矩阵中心值
#             pic_value[4][4] = pic[loc_id]
            
#             for k in range(4):
#                 # 填充灰度矩阵第5行前四个值与后四个值
#                 pic_value[4][k] = counts_gray[day][(4+loc_id//16) * 24+loc_id % 16 + k][time_interval]
#                 pic_value[4][5+k] = counts_gray[day][(4+loc_id//16) * 24+loc_id % 16 + 5 + k][time_interval]
#                 # 填充灰度矩阵前4行与后四行
#                 for i in range(9):
#                     pic_value[k][i] = counts_gray[day][(k+loc_id//16) * 24+loc_id % 16 + i][time_interval]
#                     pic_value[5+k][i] = counts_gray[day][(5+k+loc_id//16) * 24+loc_id % 16 + i][time_interval]
#             pic_value=np.reshape(pic_value,(81,))

#             # 保存每天每个时段每个地点的像素矩阵值，拉成一行81个元素
#             pics.append(pic_value)
#             pixels = ""
#             for s in range(len(pic_value)):
#                 pixels = pixels + str(pic_value[s])+" "

#             if loc_id+1 < 10:
#                 l = "00" + str(loc_id+1)
#             elif loc_id+1 >= 10 and loc_id+1 < 100:
#                 l = "0" + str(loc_id+1)
#             else:
#                 l = str(loc_id+1)

#             if day < 10:
#                 d = "0" + str(day)
#             else:
#                 d = str(day)

#             if time_interval < 10:
#                 t = "0" + str(time_interval)
#             else:
#                 t = str(time_interval)
#             with codecs.open(PIXEL_PATH + l + '_' + d + '_' + t, 'w',
#                              'utf-8') as p:
#                 p.write(pixels)
#                 print(PIXEL_PATH + l + '_' + d + '_' + t+' is done')

print(counts.shape)
print(max_value)
print(maxvalue_index, index//600+1, index%600)


