from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def main():
    iris = load_iris()
    x, y = iris.data, iris.target
    pca_3 = PCA(n_components=3)  # 降为3维
    pca_2 = PCA(n_components=2)  # 降为2维
    x_3 = pca_3.fit_transform(x)
    x_2 = pca_2.fit_transform(x)

    # 打印降维后各个维度在数据中的重要性
    print(pca_3.explained_variance_ratio_)
    print(pca_2.explained_variance_ratio_)

    # 绘制3d图像
    plot3d = mplot3d.Axes3D(plt.figure(figsize=(4, 3)))
    plot3d.scatter(x_3[:, 0], x_3[:, 1], x_3[:, 2], c=y, s=60, alpha=0.5)
    plt.show()

    plt.scatter(x_2[:, 0], x_2[:, 1], c=y, s=60, alpha=0.5)
    plt.show()


if __name__ == "__main__":
    main()