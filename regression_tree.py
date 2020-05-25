import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression


def regression_tree():
    x = np.arange(0, 10, 0.5)
    y = [i+np.random.randn() for i in x]  # 对y=x直线增加噪音
    X = x.reshape(-1, 1)

    model1 = DecisionTreeRegressor(max_depth=1)
    model2 = DecisionTreeRegressor(max_depth=3)
    model3 = LinearRegression()

    model1.fit(X, y)
    model2.fit(X, y)
    model3.fit(X, y)

    a = np.arange(0, 10, 0.01).reshape(-1, 1)
    predict1 = model1.predict(a)
    predict2 = model2.predict(a)
    predict3 = model3.predict(a)

    plt.clf()
    plt.title("The Rregression Lines")
    plt.xlabel("data")
    plt.ylabel("predict")
    plt.scatter(x, y, label="data", color="orange", edgecolors='black')
    plt.plot(a, predict1, label="Depth=1", color="blue", linewidth=3)
    plt.plot(a, predict2, label="Depth=3", color="green", linewidth=3)
    plt.plot(a, predict3, label="Linear", color="red", linewidth=3)
    plt.legend()  # 添加图例
    plt.show()


if __name__ == "__main__":
    regression_tree()
