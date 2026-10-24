import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def my_pca(X, n_components):
    #Step1 数据中心化
    mean = np.mean(X, axis=0)   # xis=0：沿着样本维度求每个特征的均值
    X_centered = X - mean

    #Step2 计算协方差矩阵
    cov_matrix = np.cov(X_centered, rowvar=False)# rowvar=False：每一行代表一个样本；一列=一个特征;分母 n-1（无偏估计）

    #Step3 特征分解
    # cov是实对称矩阵，用eigh，比eig更稳定
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    #Step4 特征值从大到小排序,eigh输出默认从小到大，所以反转
    idx_sorted = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[idx_sorted]
    eigenvectors_sorted = eigenvectors[:, idx_sorted]

    # 选取前k个主成分
    W = eigenvectors_sorted[:, :n_components]

    #Step5 投影
    X_proj = np.dot(X_centered, W)

    return X_proj, W, eigenvalues_sorted

# ====================== 测试 & 和sklearn对比 ======================
if __name__ == "__main__":

    # 生成随机测试数据：100样本，5维特征
    np.random.seed(42)
    X = np.random.randn(100, 5)

    # 1. 手写PCA
    X_my_pca, W, eig_vals = my_pca(X, n_components=2)
    print("手写PCA降维结果 shape：", X_my_pca.shape)

    # 2. sklearn PCA（用来对齐校验）
    pca_sk = PCA(n_components=2)
    X_sk_pca = pca_sk.fit_transform(X)
    print("sklearn PCA降维结果 shape：", X_sk_pca.shape)
    print("原始数据（5维）前5个样本:\n",X[:5])
    print('手写PCA降维后（2维）前5个样本\n',X_my_pca[:5])
    print("sklearn PCA降维后（2维）前5个样本：\n", X_sk_pca[:5])
    #方差解释率
    # 手写PCA 方差解释率
    explained_variance_ratio_my = eig_vals / np.sum(eig_vals)
    print("手写PCA 前2主成分方差解释率：", explained_variance_ratio_my[:2])

    # sklearn 方差解释率
    print("sklearn PCA 前2主成分方差解释率：", pca_sk.explained_variance_ratio_)

    #绘图代码
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(X_my_pca[:, 0], X_my_pca[:, 1], label="手写PCA")
    plt.title("手写PCA降维结果")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(X_sk_pca[:, 0], X_sk_pca[:, 1], label="sklearn PCA", c="orange")
    plt.title("sklearn PCA降维结果")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

