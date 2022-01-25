import numpy as np
import pandas as pd
import os

path = os.path.dirname(__file__)

date_local1 = pd.read_excel(path + '/ElectricDate/台区标识1000967005.xlsx')
date_local2 = pd.read_excel(path + '/ElectricDate/台区标识1000970901.xlsx')

print("date_local.shape: ", date_local1.shape, date_local2.shape)

def Dateloader(DateFrame, axis_in_date, input_size):
    date = DateFrame.values[:,axis_in_date]
    date = date.reshape(-1, 365)
    train_date = np.array([[]]*input_size).T
    test_date = np.array([[]]*input_size).T
    # print(date[:, 1:1+10].shape)
    split_point = int(365*0.8)
    for i in range(0, split_point-input_size+1):
        # print(i, date[:, i:i+input_size].shape)
        train_date = np.vstack([train_date, date[:, i:i+input_size]])
    
    for i in range(split_point, 365-input_size+1):
        # print(i, date[:, i:i+input_size].shape)
        test_date = np.vstack([test_date, date[:, i:i+input_size]])
    
    # train_date = date_time_window_split[0:int(date_time_window_split.shape[0]*e0.8), :]
    # test_date = date_time_window_split[int(date_time_window_split.shape[0]*0.8):, :]
    
    print("train_date.shape: ", train_date.shape, "test_date.shape: ", test_date.shape)

    return train_date, test_date


traindate_1_1, testdate_1_1 = Dateloader(date_local1, 1, 10)
# traindate_1_2, testdate_1_2 = Dateloader(date_local1, 2, 10)
# traindate_1_3, testdate_1_3 = Dateloader(date_local1, 3, 10)
# traindate_1_4, testdate_1_4 = Dateloader(date_local1, 4, 10)

# traindate_2_1, testdate_2_1 = Dateloader(date_local2, 1, 10)
# traindate_2_2, testdate_2_2 = Dateloader(date_local2, 2, 10)
# traindate_2_3, testdate_2_3 = Dateloader(date_local2, 3, 10)
# traindate_2_4, testdate_2_4 = Dateloader(date_local2, 4, 10)

# print(testdate_1_1[0])
# print(testdate_1_1[-1])


# 从时间窗口划分后的数据还原
def redate_local(date, input_size):
    num_user = int(date.shape[0] / (int(365*0.8)-input_size+1))
    origin_date = np.array([[]]*num_user)
    # print("origin_date.shape: ", origin_date.shape)

    # print(num_user)
    for i in range(0, int(365*0.8)-input_size+1):
        origin_date = np.hstack([origin_date, date[i*num_user:(i+1)*num_user,:]])
    # print("origin_date.shape: ",origin_date.shape)

    date_ = origin_date[:, 0:input_size]
    # print(date_.shape)
    for i in range(2*input_size-1, origin_date.shape[1], input_size):
        date_ = np.hstack([date_, origin_date[:, i:i+1]])
    # print(date_.shape)
    # print(date_[0])

    date_ = date_.reshape(-1, 1)
    return date_
