import numpy as np
from sklearn import preprocessing
import warnings

warnings.filterwarnings("ignore")

x = np.array([[1, -1, 2],
              [2, 0, 0],
              [0, 1, -3]])

# # 标准化
# a = preprocessing.scale(x)
# print(a)
# print(a.mean(0), a.std(0))
#
# # 标准化（常用）
# scaler = preprocessing.StandardScaler()
# b = scaler.fit_transform(x)
# print(b)
# print(b.mean(0), b.std(0))

# # minmax
# scaler = preprocessing.MinMaxScaler()
# c = scaler.fit_transform(x)
# print(c)
# print(c.mean(0), c.std(0))

# # MaxABsScaler
# scaler = preprocessing.MaxAbsScaler()
# d = scaler.fit_transform(x)
# print(d)
# print(d.mean(0), d.std(0))

# # RobustScaler
# scaler = preprocessing.RobustScaler()
# e = scaler.fit_transform(x)
# print(e)
# print(e.mean(0), e.std(0))
#
# # Normalizer 归一化
# scaler = preprocessing.Normalizer()
# f = scaler.fit_transform(x)
# print(f)
# print(f.mean(0), f.std(0))

# # 二值化
# scaler = preprocessing.Binarizer(threshold=1)
# g = scaler.fit_transform(x)
# print(g)
# print(g.mean(0), g.std(0))
#
# # one-hot 独热编码
# oh = preprocessing.OneHotEncoder(n_values=3, sparse=False)
# h = oh.fit_transform([[0], [1], [2], [1]])
# print(h)

# # 缺失数据
# # 直接填补
# imp = preprocessing.Imputer(missing_values="NaN", strategy="mean")
# i = imp.fit_transform([[2, 8], [5, "NaN"], [6, 7]])
# print(i)
#
# # 通过学习填补
# imp = preprocessing.Imputer(missing_values="NaN", strategy="mean")
# imp.fit([[1, 2], [2, 3], [4, 5]])
# j = imp.transform([[2, 3], [5, "NaN"], [6, 7]])
# print(j)
