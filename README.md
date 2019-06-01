# DMVST_Net
包括部分数据预处理以及基于Tensorflow的DMVST_Net模型的实现，数据用的是2016年1月份纽约市的数据

# 原始数据

- <a href="https://data.cityofnewyork.us/browse?q=taxi&page=1" target="_blank">出租车数据下载地址</a>，我用的是2016 Green Taxi Trip Data数据

- <a href="http://www.meteomanz.com/index?l=1&cou=4030&ind=72506&ty=hp&d1=01&m1=01&y1=2016&h1=00Z&d2=31&m2=01&y2=2016&h2=23Z" target="_blank">天气数据下载地址</a>，我用的是2016年1月份纽约中央公园的天气数据

- <a href="https://aqs.epa.gov/aqsweb/airdata/download_files.html#AQI" target="_blank">空气数据下载地址</a>，我用的是daily_aqi_by_county_2016数据

# 数据预处理
- Spatial_longitude_latitude.py:将纽约市按照经纬度划分为不同的地区
- Spatial_time_id_to_txt.py:将对应预测地区在某一时刻的9*9需求像素矩阵拉成为81维向量存储在一个文件中
- Temporal_weather_condition.py:处理纽约市的天气状况，不同的天气对应不同的数字
>  'Clear' :1</br>
   'Scattered clouds':2</br>
   'Few clouds':3</br>
   'Cloudy':4</br>
   'Overcast':5</br>
   'Few clouds, mnist':6</br>
   'fog':7</br>
   'rain':8</br>
   'snow':9</br>
   'Overcast, mist':10</br>
   'Overcast, rain':11</br>
   'Overcast, snow':12</br>
- Temporal_aq_data.py:处理纽约市的空气质量数据
- Temporal_time_data.py:对应时刻是在一星期中的哪天，该天是否为节假日
- Temporal_context_data_merge.py:将天气、空气质量、星期几、是否为节假日信息合并在一起作为时间视图中每个时刻输入的上下文数据
- Semantic_demand_pattern.py:确定每个预测地区的需求模式
- Semantic_weighted_graph.py：确定需求权重图中的权重
- Semantic_embedding.py：用LINE方法生成语义向量（C++文件为LINE方法的实现代码，<a href="https://github.com/tangjianpku/LINE" target="_blank">LINE方法源码地址</a>
# 模型实现与数据集
- DMVST_TFReacord.py:生成数据集
- start.py:启动程序，运行该文件开始模型的训练
- DMVST_Net_CNN.py:局部CNN的实现
- DMVST_Net_LSTM,py:时间视图的实现
- DMVST_Net_Semantic.py:语义视图的实现
- DMVST_Net_forward.py:DMVST_Net模型的实现
- DMVST_Net_backward.py:反向传播过程的实现
# 代码使用
- 把原始数据文件下载到根目录
- 把数据预处理文件按照之前的介绍顺序依次执行（处理时间比较长，其中生成语义向量文件程序可以中途停止生成，之后只要重启Semantic_embedding.py就可以从上次生成的位置继续生成）
- 运行DMVST_TFReacord.py，生成数据集
- 运行start.py开始模型训练（训练过程中会不断写入两个文件records1、verifying_resault1，前者记录每喂入10个batch后模型的一些信息，后者记录验证时的信息）
