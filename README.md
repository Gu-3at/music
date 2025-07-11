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

# 加载数据
df = pd.read_csv('nigerian-songs.csv')

# 显示数据的基本信息
print("数据基本信息：")
print(df.info())

# 显示数据的前几行
print("\n数据的前几行：")
print(df.head())

# 描述性统计分析
print("\n描述性统计分析：")
print(df.describe())

# 按艺术家分组，计算每种音乐类型的平均流行度
print("\n按艺术家分组，计算每种音乐类型的平均流行度：")
artist_genre_popularity = df.groupby(['artist', 'artist_top_genre'])['popularity'].mean()
print(artist_genre_popularity)

# 按音乐类型分组，计算每种音乐类型的平均时长
print("\n按音乐类型分组，计算每种音乐类型的平均时长：")
genre_length = df.groupby('artist_top_genre')['length'].mean()
print(genre_length)

# 按年份分组，计算每年发布的歌曲数量
# 确保 release_date 列是字符串类型
df['release_date'] = df['release_date'].astype(str)
year_song_count = df.groupby(df['release_date'].str[:4])['name'].count()
print("\n按年份分组，计算每年发布的歌曲数量：")
print(year_song_count)

# 绘制流行度的直方图
import matplotlib.pyplot as plt
df['popularity'].plot(kind='hist', bins=20, title='歌曲流行度分布')
plt.xlabel('流行度')
plt.ylabel('歌曲数量')
plt.show()

# 绘制不同音乐类型流行度的箱线图
df.boxplot(column='popularity', by='artist_top_genre', grid=True, figsize=(12, 8))
plt.title('不同音乐类型流行度分布')
plt.xlabel('音乐类型')
plt.ylabel('流行度')
plt.suptitle("")  # 隐藏自动生成的标题
plt.show()
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

t - SNE的主要目标是在低维空间中保留高维数据点之间的相似性。它通过计算高维空间中数据点之间的相似度，并在低维空间中尽量保持这种相似度来实现降维。

应用场景

    数据可视化
        t - SNE广泛应用于数据可视化领域，特别是对于高维数据集。例如，在图像识别中，它可以将图像特征映射到二维或三维空间，帮助研究人员直观地观察不同类别图像之间的分布情况；在生物信息学中，它可以用于可视化基因表达数据，发现基因之间的相似性和簇结构。
    聚类分析
        由于t - SNE能够发现数据中的簇结构，因此它可以作为聚类分析的预处理步骤。通过t - SNE降维后，可以使用传统的聚类算法（如K - Means）对低维数据进行聚类，从而提高聚类效果。
    异常检测
        t - SNE可以用于异常检测。在低维嵌入中，异常数据点通常会与其他数据点分开，形成孤立的点或小簇。通过观察低维嵌入中的数据分布，可以发现异常数据点，从而实现异常检测。

参数选择

    困惑度（Perplexity）
        困惑度是t - SNE的一个重要参数，它影响着降维结果的局部和全局结构。困惑度越大，每个数据点的邻居数量越多，降维结果越倾向于保留全局结构；困惑度越小，每个数据点的邻居数量越少，降维结果越倾向于保留局部结构。通常，困惑度的取值范围在5 - 50之间，具体取值需要根据数据集的大小和分布情况进行调整。
    学习率（Learning Rate）
        学习率控制着优化过程中的步长。学习率过高可能导致优化过程不稳定，数据点在低维空间中跳动过大；学习率过低可能导致优化过程收敛过慢。通常，学习率的取值范围在100 - 1000之间。
    迭代次数（Max Iterations）
        迭代次数决定了优化过程的运行时间。迭代次数越多，优化过程越充分，但运行时间也越长。通常，迭代次数的取值范围在500 - 2000之间，具体取值需要根据数据集的大小和复杂程度进行调整。
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
KMeans Silhouette Score: 0.23387880286414295
Agglomerative Clustering Silhouette Score: 0.14241639056082078
KMeans Davies-Bouldin Score: 1.673985763083236
Agglomerative Clustering Davies-Bouldin Score: 1.5164838858018026
## 结论
从轮廓系数来看，两种聚类算法的效果都不理想，KMeans稍好于层次聚类。从戴维斯 - 丁指数来看，层次聚类相对优于KMeans，但差距不大。这可能是因为数据本身的特性不适合这两种聚类算法。例如，数据可能存在噪声、维度较高或者分布形状复杂等情况。在这种情况下，可能需要考虑其他聚类算法，如DBSCAN（对于有噪声和形状不规则的数据有优势）或者高斯混合模型（对于簇形状为椭圆形的数据效果较好）等，或者对数据进行预处理，如降维、去除噪声等操作后再进行聚类。
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
### K-means是否对数据归一敏感：
这个我注意到了。如果数据的尺度不一样，聚类结果可能会受影响。比如，如果一个特征的值特别大，它可能会主导聚类结果。所以，我们可能需要先对数据进行归一化处理。
### 特征分析：
我觉得特征分析很重要。我们可以通过相关性分析来找出哪些特征是相关的，然后去掉一些冗余的特征。还有，PCA也是个好方法，可以帮我们降维，同时保留最重要的信息。
### 理解PCA（主成分分析）：
PCA它主要是把高维数据降到低维，同时尽量保留数据的变异性。我们先要对数据进行标准化，然后计算协方差矩阵，找出最重要的几个主成分，最后把数据投影到这些主成分上。
### k-Means 聚类效果分析
我们可以用一些指标来评估k-Means的聚类效果，比如轮廓系数、Calinski-Harabasz指数和Davies-Bouldin指数。这些指标的值越大或越小，通常表示聚类效果越好。另外，我们还可以画图看看，直观地感受聚类效果。
### k-Means 初始值敏感分析
我发现k-Means对初始质心的选择挺敏感的，不同的初始值可能会导致不同的聚类结果。为了减少这种影响，我们可以多运行几次k-Means，每次都用不同的初始质心，然后选一个效果最好的结果。或者用k-Means++这种改进的初始化方法，它能让我们少操点心。
### k-Means 数据归一敏感性分析
k-Means对数据的尺度挺敏感的，因为距离计算会受到数据尺度的影响。为了减少这种影响，我们通常需要对数据进行归一化或标准化处理，让不同特征对距离计算的影响更均衡，这样聚类效果会更好。
### k-Means 是否适合处理高维数据处理缺失值？
对于高维数据，k-Means可能会遇到“维度灾难”，就是距离计算变得不那么有效。我们可以考虑先用PCA之类的降维技术减少维度，然后再用k-Means。至于缺失值，k-Means本身不太能处理，我们可能需要先删除含有缺失值的样本，或者用均值、中位数之类的方法填充缺失值。