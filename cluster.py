import warnings
from sklearn.datasets import samples_generator
from sklearn import cluster, metrics
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


def main():
    # 过滤警告
    warnings.filterwarnings("ignore")

    # 创建“点滴”数据
    # x, y = samples_generator.make_blobs(n_samples=200, centers=2, cluster_std=1, random_state=0)
    # 创建“月牙”数据
    # x, y = samples_generator.make_moons(n_samples=200, noise=0.05, random_state=0)
    # 创建“环形”数据
    x, y = samples_generator.make_circles(n_samples=200, noise=0.05, random_state=0, factor=0.4)

    """
    创建七种聚类算法
    """
    # clusters = cluster.KMeans(2)  # K-means++
    # clusters = cluster.MeanShift()  # 均值迁移
    # clusters = cluster. AgglomerativeClustering(2)  # 层聚类
    # clusters = cluster.AffinityPropagation()  # AP聚类
    # clusters = cluster.SpectralClustering(n_clusters=2, affinity="nearest_neighbors")  # 谱聚类
    # clusters = cluster.DBSCAN(eps=0.55, min_samples=5)  # 密度聚类
    clusters = GaussianMixture(n_components=2)  # 高斯分布

    # 拟合
    _x = clusters.fit_predict(x)

    """
    三种评价方法
    """
    # 1.轮廓系数
    print(metrics.silhouette_score(x, _x))
    # 2.CH指数
    print(metrics.calinski_harabasz_score(x, _x))
    # 3.戴维森堡丁指数
    print(metrics.davies_bouldin_score(x, _x))

    # 绘图
    plt.scatter(x[:, 0], x[:, 1], c=_x, cmap='viridis')
    plt.show()


if __name__ == "__main__":
    main()