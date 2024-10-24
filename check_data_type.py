# import pickle

# 查看label内容和格式
# file_path_label = 'dataset/CMUMOSI/CMUMOSI_features_raw_2way.pkl'
#
# with open(file_path_label, 'rb') as file:
#     data = pickle.load(file)

# print(type(data)) #总体数据集是list
# print(len(data)) # data的list中有七个元素
# for i in range(len(data)): # 前四个是字典后三个是set
#     print(type(data[i]))
#     print("\n")

# for i in range(len(data)):
#     if isinstance(data[i], dict):
#         print((data[i]).keys())
#     else:
#         print((data[i]))
#     print(len(data[i])) # 长度为93 93 93 93 53 10 31
#     print()
# data[0] 视频编号：切片（list）
# data[1] 视频编号：每个切片的label（情感强度标签）
# data[2] 视频编号：‘’说话人（mosi没有）
# data[3] 视频编号：text
# data[4] 训练集视频编号
# data[5] 评估集视频编号
# data[6] 测试集视频编号


# # 查看feature
# import numpy as np
# # 文件路径
# file_path = 'dataset/CMUMOSI/features/wav2vec-large-c-UTT/BI97DNYfe5I_5.npy'
# # 加载 .npy 文件
# feature_text_1_1 = np.load(file_path)
#
# # 查看特征的内容和类型
# print(type(feature_text_1_1))  # 类型 numpy.ndarray
# print(feature_text_1_1.shape)  # 数组形状 (1024,) 一维
# print(feature_text_1_1.dtype)  # 元素类型 float32
# print(feature_text_1_1.size)   # 数组长度 1024
# # 音频数组长度是512