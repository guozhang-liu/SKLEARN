import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge

def kernelridge():
    # 制作数据
    x = np.random.randn(100, 1)*3 + 7
    y = np.sin(x).ravel()

    # 制造噪点
    y[::5] += np.random.randn(x.shape[0]//5)

    # 构建模型
    kr = GridSearchCV(KernelRidge(),
                      param_grid={"kernel": ["sigmoid", "rbf", "laplacian", "polynomial"],
                                  "gamma": np.logspace(-2, 2, 5),
                                  "alpha": [1e0, 1e-1, 1e-2, 1e-3]
                                  })

    # 拟合
    kr.fit(x, y)

    # 预测
    a = np.linspace(0, 7, 100)
    b = kr.predict(a.reshape(-1, 1))
    print(kr.best_score_, kr.best_params_)

    # 绘图
    plt.scatter(x, y)
    plt.plot(a, b, linewidth=3, color="red")
    plt.show()


if __name__ == "__main__":
    kernelridge()