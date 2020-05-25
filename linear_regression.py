import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import sklearn.linear_model as model
from sklearn import metrics
import matplotlib.pyplot as plt


def regression_train():
    # 定变量
    np.random.RandomState()

    # 制作数据集
    dataset = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=10)

    # 划分数据集
    x, y = dataset[0], dataset[1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # 构建模型
    # reg = model.LinearRegression()
    # reg = model.Ridge()
    # reg = model.Lasso()
    # reg = model.ElasticNet()
    reg = model.BayesianRidge()

    # 拟合
    reg.fit(x_train, y_train)
    # print(reg.coef_, reg.intercept_)

    y_predict = reg.predict(x_test)

    # 回归评估指标
    # 平均绝对误差
    print(metrics.mean_absolute_error(y_test, y_predict))
    # 均方误差
    print(metrics.mean_squared_error(y_test, y_predict))
    # R2
    print(metrics.r2_score(y_test, y_predict))
    # 可解释方差
    print(metrics.explained_variance_score(y_test, y_predict))

    # 绘图
    _x = np.array([-2.5, 2.5])
    _y = reg.predict(_x[:, None])

    plt.scatter(x_test, y_test, color="red")
    plt.plot(_x, _y, linewidth=3, color="orange")
    plt.show()


if __name__ == "__main__":
    regression_train()