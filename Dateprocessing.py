import pandas as pd
import glob
import os

path = os.path.dirname(__file__)
file_list = glob.glob(path + '/ElectricDate/*.xlsx')

# date_total = pd.DataFrame(columns=['表资产', '终端类型', '台区标识', 
#                             '正向有功总电量', '正向有功尖电量', '正向有功峰电量', '正向有功谷电量', '正向有功平电量', '统计时间'])

# 合并Excel
date_length = []
date_list = []
for file_name in file_list:
    date_per_month = pd.read_excel(file_name, converters={'台区标识': str})
    date_per_length = len(date_per_month)
    # print(file_name, "格式为: ", date_per_month.shape)
    # print(file_name, "长度为: ", date_per_length)
    date_list.append(date_per_month)
    date_length.append(date_per_length)

date_total = pd.concat(date_list, ignore_index=True, axis=0)
print(date_total.shape)
# print(date_list)


# 选择两个台区
date_local1 = date_total[(date_total['台区标识'] == '1000967005') & (date_total['终端类型'] == '集中器')]
date_local2 = date_total[(date_total['台区标识'] == '1000970901') & (date_total['终端类型'] == '集中器')]


# 过滤掉台区中数据量不等于365的电表
count_per_date1 = date_local1[['表资产', '统计时间']].groupby('表资产').size().reset_index(name='count') # 计算每个表资产的数据量
tableName_list1 = count_per_date1[count_per_date1['count'] == 365]['表资产'].values.tolist() # 表资产list
date_local1 = date_local1[date_local1['表资产'].isin(tableName_list1)]

count_per_date2 = date_local2[['表资产', '统计时间']].groupby('表资产').size().reset_index(name='count') # 计算每个表资产的数据量
tableName_list2 = count_per_date2[count_per_date2['count'] == 365]['表资产'].values.tolist() # 表资产list
date_local2 = date_local2[date_local2['表资产'].isin(tableName_list2)]


# 缺失值填充0
date_local1.fillna(value=0, inplace=True)
date_local2.fillna(value=0, inplace=True)


# 数据标准化 
def date_normalize(dateframe):
    date1 = dateframe[['正向有功总电量', '正向有功尖电量', '正向有功峰电量', '正向有功谷电量', '正向有功平电量']]
    dateframe[['正向有功总电量', '正向有功尖电量', '正向有功峰电量', '正向有功谷电量', '正向有功平电量']] = (date1 - date1.mean()) / date1.std()
    return dateframe

date_local1_afternorm = date_normalize(date_local1)
date_local2_afternorm = date_normalize(date_local2)
# print(date_local1_afternorm.shape)


# 每一个表资产放在一起，并按时间升序
date_local1_afternorm.sort_values(by=['表资产', '统计时间'], ascending=[True, True], inplace=True)
date_local2_afternorm.sort_values(by=['表资产', '统计时间'], ascending=[True, True], inplace=True)

# 输出处理后数据，正向有功尖电量全为0，舍弃该字段
final_date_local1 = date_local1_afternorm[['表资产', '正向有功总电量', '正向有功峰电量', '正向有功谷电量', '正向有功平电量', '统计时间']]
final_date_local2 = date_local2_afternorm[['表资产', '正向有功总电量', '正向有功峰电量', '正向有功谷电量', '正向有功平电量', '统计时间']]

final_date_local1.to_excel(path + '/ElectricDate/台区标识1000967005.xlsx', index=False)
final_date_local2.to_excel(path + '/ElectricDate/台区标识1000970901.xlsx', index=False)

