import numpy as np
from sklearn import datasets, preprocessing, neighbors
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def iris_train():
    # 定变量
    np.random.RandomState()

    # 加载数据
    iris = datasets.load_iris()
    x, y = iris.data, iris.target

    # 拆分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # 数据预处理
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # 创建模型
    knn = neighbors.KNeighborsClassifier(n_neighbors=12)

    # 拟合
    knn.fit(x_train, y_train)

    # 交叉验证
    scores = cross_val_score(knn, x_train, y_train, cv=8, scoring="accuracy")
    print(scores)
    print(scores.mean())

    # 评估
    y_predict = knn.predict(x_test)

    print(accuracy_score(y_predict, y_test))


def best_paras():

    iris = datasets.load_iris()
    x, y = iris.data, iris.target

    para = np.arange(1, 31)
    score = []
    for i in para:
        knn = neighbors.KNeighborsClassifier(n_neighbors=i)
        knn.fit(x, y)
        scores = cross_val_score(knn, x, y, cv=5, scoring="accuracy")
        score.append(scores.mean())
    plt.figure()
    plt.plot(para, score)
    plt.show()


if __name__ == "__main__":
    best_paras()