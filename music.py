# 导入必要的库
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
# 消除警告
plt.rcParams['font.sans-serif'] = ['SimHei']

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
# 选择特征

features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'tempo']

# 查看缺失值情况
missing_values = df[features].isnull().sum()
print("Missing values per feature:\n", missing_values)

# 根据缺失值情况决定处理方式
# 如果缺失值较少，可以选择删除包含缺失值的行
if missing_values.sum() > 0:
    print("Dropping rows with missing values.")
    df.dropna(subset=features, inplace=True)
else:
    print("No missing values found.")

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# 确定最佳聚类数（KMeans）
def plot_elbow_method(X_scaled):
    distortions = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 10), distortions, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()

# 确定最佳聚类数（轮廓系数）
def plot_silhouette_scores(X_scaled):
    silhouette_scores = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, labels))

    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 10), silhouette_scores, marker='o')
    plt.title('Silhouette Score')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.show()

# 调用函数绘制肘部法则和轮廓系数图
plot_elbow_method(X_scaled)
plot_silhouette_scores(X_scaled)

# 根据肘部法则和轮廓系数选择最佳聚类数，这里假设最佳聚类数为3
n_clusters = 3

# KMeans聚类
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# 层次聚类（Agglomerative Clustering）
agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
agg_labels = agg_clustering.fit_predict(X_scaled)

# 添加聚类标签到原始数据
df['KMeans_Cluster'] = kmeans_labels
df['Agg_Cluster'] = agg_labels

# 使用t-SNE进行降维以便于可视化
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# 可视化聚类结果的函数
def plot_clustering_results(X_tsne, labels, title):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, palette='viridis', legend='full')
    plt.title(title)
    plt.show()

# 可视化KMeans聚类结果
plot_clustering_results(X_tsne, df['KMeans_Cluster'], 'KMeans Clustering with t-SNE')

# 可视化层次聚类结果
plot_clustering_results(X_tsne, df['Agg_Cluster'], 'Agglomerative Clustering with t-SNE')

# 评估聚类质量
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
agg_silhouette = silhouette_score(X_scaled, agg_labels)
kmeans_db_score = davies_bouldin_score(X_scaled, kmeans_labels)
agg_db_score = davies_bouldin_score(X_scaled, agg_labels)

print(f'KMeans Silhouette Score: {kmeans_silhouette}')
print(f'Agglomerative Clustering Silhouette Score: {agg_silhouette}')
print(f'KMeans Davies-Bouldin Score: {kmeans_db_score}')
print(f'Agglomerative Clustering Davies-Bouldin Score: {agg_db_score}')

# 特征分布直方图
df[features].hist(bins=20, figsize=(15, 10), layout=(3, 3))
plt.suptitle('Feature Distribution Histograms', fontsize=16)
plt.show()

# 特征之间的相关性热图
correlation_matrix = df[features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.show()

# 特征之间的散点图矩阵
sns.pairplot(df[features], diag_kind='kde', plot_kws={'alpha': 0.5})
plt.suptitle('Feature Scatter Plot Matrix', fontsize=16)
plt.show()

# 特征的箱线图
plt.figure(figsize=(15, 10))
df[features].boxplot()
plt.title('Feature Boxplots')
plt.show()

