import numpy as np
from sklearn import ensemble, neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.RandomState(0)

# 加载数据
wine = datasets.load_wine()

# 划分训练集与测试集
x, y = wine.data, wine.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# 数据预处理
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 创建模型
clf = ensemble.AdaBoostClassifier(n_estimators=50)

# 模型拟合
clf.fit(x_train, y_train)

# 预测
y_pred = clf.predict(x_test)

# 评估
print(accuracy_score(y_test, y_pred))