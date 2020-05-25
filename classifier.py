from sklearn import datasets, preprocessing, svm, linear_model, neighbors
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def classifier_structure():
    warnings.filterwarnings("ignore")  # sklearn因为版本问题会有一些警告，此处可以过滤警告
    np.random.RandomState(0)  # 定参数

    #  导入数据
    iris = datasets.load_iris()
    x, y = iris.data, iris.target

    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # 数据预处理
    scaler = preprocessing.StandardScaler().fit(x)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # 创建模型
    clf = neighbors.KNeighborsClassifier(n_neighbors=12, algorithm="kd_tree")  # 近邻算法kd_tree
    # clf = linear_model.SGDClassifier()  # 随机梯度下降法
    # clf = linear_model.LogisticRegression()  # 逻辑斯蒂回归用于分类问题
    # clf = svm.SVC(kernel="rbf")  # 支持向量机

    # 拟合
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)

    # 评估
    print(accuracy_score(y_test, y_predict))  # 精确度
    print(f1_score(y_test, y_predict, average='micro'))  # f1评分
    print(classification_report(y_test, y_predict))  # 分类报告
    print(confusion_matrix(y_test, y_predict))  # 混淆矩阵


if __name__ == "__main__":
    classifier_structure()
