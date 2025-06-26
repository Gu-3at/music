# Nigerian Songs 聚类分析
## 项目概述
本项目旨在对Nigerian Songs数据集进行聚类分析，通过多种音频特征来发现歌曲之间的模式，并将相似的歌曲分组到一起。
目录

    项目概述
    安装
    数据加载与预处理
    聚类分析
    可视化
    评估
    结论
    特征分布与相关性分析

## 安装
要运行本项目，你需要安装以下库：

    pandas
    scikit-learn
    matplotlib
    seaborn

你可以使用pip进行安装：
pip install pandas scikit-learn matplotlib seaborn
## 数据加载与预处理
本项目使用的数据集是“nigerian-songs.csv”。以下是用于聚类分析所选择的特征：

    舞曲度（danceability）
    能量（energy）
    响度（loudness）
    语言性（speechiness)
    原声性（acousticness）
    乐器性（instrumentalness）
    现场感（liveness）
    节奏（tempo）

以下是加载数据集、检查缺失值并处理它们的代码片段。然后使用StandardScaler对选定的特征进行标准化。
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 加载数据集
df = pd.read_csv('nigerian-songs.csv')

# 选择特征
features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'tempo']

# 检查缺失值
missing_values = df[features].isnull().sum()
print("每个特征的缺失值数量：\n", missing_values)

# 处理缺失值
if missing_values.sum() > 0:
    print("删除包含缺失值的行。")
    df.dropna(subset=features, inplace=True)
else:
    print("未发现缺失值。")

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
```
## 聚类分析
确定最佳聚类数
使用以下两种方法确定最佳聚类数：

    肘部法则（Elbow Method）
    轮廓系数（Silhouette Score）

以下是绘制肘部法则和轮廓系数图的代码片段，以帮助可视化最佳聚类数。
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 肘部法则
def plot_elbow_method(X_scaled):
    distortions = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 10), distortions, marker='o')
    plt.title('肘部法则')
    plt.xlabel('聚类数')
    plt.ylabel('失真度')
    plt.show()

plot_elbow_method(X_scaled)

# 轮廓系数
from sklearn.metrics import silhouette_score

def plot_silhouette_scores(X_scaled):
    silhouette_scores = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, labels))

    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 10), silhouette_scores, marker='o')
    plt.title('轮廓系数')
    plt.xlabel('聚类数')
    plt.ylabel('轮廓系数')
    plt.show()

plot_silhouette_scores(X_scaled)
```
## 聚类算法
应用以下两种聚类算法：

    KMeans
    层次聚类（Agglomerative Clustering）

以下是使用KMeans和层次聚类进行聚类的代码片段，假设最佳聚类数为3。
```python
# KMeans聚类
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# 层次聚类
from sklearn.cluster import AgglomerativeClustering

agg_clustering = AgglomerativeClustering(n_clusters=3)
agg_labels = agg_clustering.fit_predict(X_scaled)

# 将聚类标签添加到原始数据集中
df['KMeans_Cluster'] = kmeans_labels
df['Agg_Cluster'] = agg_labels
```
## 可视化
t-SNE降维
使用t-SNE对数据进行降维，以便于可视化。
```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
```
## 聚类结果可视化
使用t-SNE降维后的数据绘制散点图，可视化聚类结果。
```python
import seaborn as sns

# 可视化函数
def plot_clustering_results(X_tsne, labels, title):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, palette='viridis', legend='full')
    plt.title(title)
    plt.show()

# 可视化KMeans聚类结果
plot_clustering_results(X_tsne, df['KMeans_Cluster'], 'KMeans聚类结果（t-SNE可视化）')

# 可视化层次聚类结果
plot_clustering_results(X_tsne, df['Agg_Cluster'], '层次聚类结果（t-SNE可视化）')
```
## 评估
使用轮廓系数和戴维斯-邦丁指数（Davies-Bouldin Score）评估聚类质量。
```python
from sklearn.metrics import silhouette_score, davies_bouldin_score

# 评估KMeans聚类
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
kmeans_db_score = davies_bouldin_score(X_scaled, kmeans_labels)

# 评估层次聚类
agg_silhouette = silhouette_score(X_scaled, agg_labels)
agg_db_score = davies_bouldin_score(X_scaled, agg_labels)

print(f'KMeans轮廓系数：{kmeans_silhouette}')
print(f'层次聚类轮廓系数：{agg_silhouette}')
print(f'KMeans戴维斯-邦丁指数：{kmeans_db_score}')
print(f'层次聚类戴维斯-邦丁指数：{agg_db_score}')
```
## 结论
本项目通过基于选定音频特征的聚类分析，对Nigerian Songs进行了分组。通过肘部法则和轮廓系数确定了最佳聚类数。应用了KMeans和层次聚类算法，并使用t-SNE对聚类结果进行了可视化。使用轮廓系数和戴维斯-邦丁指数评估了聚类质量。可以进一步进行分析，以深入了解不同歌曲聚类的模式和特征。
# 特征分布与相关性分析
## 特征分布直方图
绘制选定特征的直方图，以可视化它们的分布。
```python
df[features].hist(bins=20, figsize=(15, 10), layout=(3, 3))
plt.suptitle('特征分布直方图', fontsize=16)
plt.show()
```
## 特征相关性热图
使用热图可视化选定特征之间的相关性。
```python
correlation_matrix = df[features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('特征相关性热图')
plt.show()
```
## 特征散点图矩阵
创建散点图矩阵，以可视化特征对之间的关系。
```python
sns.pairplot(df[features], diag_kind='kde', plot_kws={'alpha': 0.5})
plt.suptitle('特征散点图矩阵', fontsize=16)
plt.show()
```
## 特征箱线图
绘制选定特征的箱线图，以可视化它们的分布并识别任何异常值。
```python
plt.figure(figsize=(15, 10))
df[features].boxplot()
plt.title('特征箱线图')
plt.show()
```