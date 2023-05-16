


import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import Normalizer, LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as shc
import matplotlib.patheffects as PathEffects


from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, make_scorer
from sklearn.model_selection import GridSearchCV
import numpy as np

# In[2]:


from resources import ConfigS3
define = ConfigS3()


# In[3]:


import config as CONFIG
import boto3


# In[4]:


# folder_path = 'Result_with_velocity'
# csv_list_file = define.bucket.objects.filter(Prefix=folder_path)


# # In[5]:


# total_df = pd.DataFrame()
# count = 0
# for obj in csv_list_file:
#     try:
#         define.download_file_from_s3(obj.key, obj.key)
#         temp_df = pd.read_csv(obj.key)
#         total_df = pd.concat([total_df, temp_df])
#     except:
#         pass
#     define.remove_local_file(obj.key)
def processing_data():
    TRAIN_DIR = os.path.join('.', 'is_hot_tomtom_segment_status_with_velocity')
    # TRAIN_DIR = os.path.join('.', 'is_hot_tomtom_segment_status3')
    csv_file_list = os.listdir(TRAIN_DIR)
    total_df = pd.DataFrame()
    for csv_file in csv_file_list:
        temp_df = pd.read_csv(os.path.join(TRAIN_DIR, csv_file))
        total_df = pd.concat([total_df, temp_df])

    total_df.drop(["Unnamed: 0"], axis=1, inplace=True)
    total_df.head(5)


    label_encoder = LabelEncoder()
    # total_df['base_LOS']= label_encoder.fit_transform(total_df['base_LOS'])
    total_df['isHot']= label_encoder.fit_transform(total_df['isHot'])
    total_df['weather']= label_encoder.fit_transform(total_df['weather'])

    return total_df


def _elbow_function(period_df):
    period_df_time = period_df.drop(["period", "weekday", "temperature", "segment_id", "isHot", "weather", "base_LOS", "district", "is_morning", "date"], axis=1)
    scaled_df = StandardScaler().fit_transform(period_df_time)
    sse = []
    # Lặp qua các giá trị k từ 1 đến 10
    for k in range(1, 11):
        # Khởi tạo model KMeans với số lượng cụm k
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        # Fit dữ liệu vào model
        kmeans.fit(scaled_df)
        # Lưu giá trị SSE vào mảng
        sse.append(kmeans.inertia_)
    
    # Vẽ biểu đồ Elbow để tìm số lượng cụm tối ưu
    plt.plot(range(1, 11), sse)
    plt.title('Phương pháp Elbow')
    plt.xlabel('Số lượng cụm')
    plt.ylabel('SSE')
    plt.show()
    
    return sse


# In[13]:


def find_optimal_k(X, is_test = False):
    # Initialize variables
    Ks = range(2, 11)
    scores_elbow = []
    scores_silhouette = []

    # Loop over different values of K
    for K in Ks:
        # Fit K-means model
        kmeans = KMeans(n_clusters=K, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)

        # Calculate distortion (elbow criterion)
        distortion = kmeans.inertia_
        scores_elbow.append(distortion)

        # Calculate Silhouette Score
        if K > 1:
            silhouette_avg = silhouette_score(X, kmeans.labels_)
            scores_silhouette.append(silhouette_avg)
        else:
            scores_silhouette.append(0)

    # Find optimal K based on elbow criterion and Silhouette Score
    scores_elbow = np.array(scores_elbow)
    scores_silhouette = np.array(scores_silhouette)
    diff = np.diff(scores_elbow)
    elbow_index = np.argmax(diff)
    silhouette_index = np.argmax(scores_silhouette)
    scores_elbow_silhouette = diff/scores_silhouette[:-1]
    n_numbers = range(2, 11)
    optimal_K = n_numbers[np.argmax(scores_elbow_silhouette)]
    
#     K = silhouette_index if abs(elbow_index - silhouette_index) <= 2 or elbow_index_2 == silhouette_index else elbow_index_1
    if abs(elbow_index - silhouette_index) <=2:
        K = elbow_index  if elbow_index > silhouette_index else silhouette_index
    else:
        K = elbow_index 
    if is_test:
        # Plot results
        plt.plot(Ks, scores_elbow, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        plt.axvline(x=optimal_K, linestyle='--', color='red', label='Optimal k (elbow)')
        plt.legend()
        plt.show()

        plt.plot(Ks, scores_silhouette, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Silhouette Score')
        plt.title('The Silhouette Score showing the optimal k')
        plt.axvline(x=optimal_K, linestyle='--', color='red', label='Optimal k (Silhouette Score)')
        plt.legend()
        plt.show()
    return optimal_K


# In[14]:


def k_mean_function(scaled_df, k_number = 4):
    kmeans = KMeans(n_clusters=k_number, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_df)
    return kmeans.labels_


# In[15]:


def _plot_kmean_scatter(X, labels):
    '''
    X: dữ liệu đầu vào
    labels: nhãn dự báo
    '''
    # lựa chọn màu sắc
    num_classes = len(np.unique(labels))
    palette = np.array(sns.color_palette("hls", num_classes))

    # vẽ biểu đồ scatter
    fig = plt.figure(figsize=(12, 10))
    ax = plt.subplot()
    sc = ax.scatter(X[:,0], X[:,1], lw=0, s=40, c=palette[labels.astype(np.int32)])

    # thêm nhãn cho mỗi cluster
    txts = []

    for i in range(num_classes):
        # Vẽ text tên cụm tại trung vị của mỗi cụm
        xtext, ytext = np.median(X[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    plt.title('t-sne visualization')


# In[16]:


def _plot_dendograms(scaled_df):
    plt.figure(figsize=(20, 7))
    plt.title("Dendograms")
    dend = shc.dendrogram(shc.linkage(scaled_df, method='ward'))
    plt.axhline(200, linestyle='--')
    plt.xlabel('sample indice')
    plt.ylabel('dissimilarity metric cluster')


# In[17]:


def AgglomerativeClustering_function(scaled_df, number_clusters = 4):
    cluster = AgglomerativeClustering(n_clusters=number_clusters, affinity='euclidean', linkage='ward')
    labels = cluster.fit_predict(scaled_df)
    return labels


# In[18]:


def DBSAN_function(X):
    std = MinMaxScaler()
    X_std = std.fit_transform(X)

    # Define range of values for epsilon and minPts
    eps_range = np.arange(0.1, 1.0, 0.1)
    minPts_range = range(1, 11)

    # Define parameters for Grid Search
    param_grid = {'eps': eps_range, 'min_samples': minPts_range}

    # Define DBSCAN model
    dbscan = DBSCAN()
    # Define custom scorer
    silhouette_scorer = make_scorer(silhouette_score)

    # Define Grid Search object
    grid_search = GridSearchCV(dbscan, param_grid=param_grid, scoring=silhouette_scorer)

    # Fit Grid Search object to data
    grid_search.fit(X_std)

    # Print best parameters and best score
    print("Best Parameters: ", grid_search.best_params_)
    print("Best Silhouette Score: ", grid_search.best_score_)
    return grid_search.best_params_


# In[19]:


def test_dbscan(X):
    std = MinMaxScaler()
    X_std = std.fit_transform(X)
    dbscan = DBSCAN(eps=0.1, min_samples=11, metric='euclidean')
    labels = dbscan.fit_predict(X_std)
    return labels



# In[21]:


def get_label_with_Agglomerative(period_df,  number_clusters = 4, is_test = False):
    period_df_time = period_df.drop(["period", "weekday", "temperature", "segment_id", "isHot", "weather", "base_LOS", "district", "is_morning", "date"], axis=1, errors = 'raise')
    if "label" in period_df_time.columns:
        period_df_time = period_df_time.drop(["label"], axis = 1)
    scaled_df = MinMaxScaler().fit_transform(period_df_time)
    _plot_dendograms(scaled_df)
    label_result = AgglomerativeClustering_function(scaled_df, number_clusters)
    if is_test:
        _plot_kmean_scatter(scaled_df, label_result)
    return label_result


# In[22]:


def get_label(period_df, is_test = False):
    period_df_time = period_df.drop(["period", "weekday", "temperature", "segment_id", "isHot", "weather", "base_LOS", "district", "is_morning", "date"], axis=1)
    if "label" in period_df_time.columns:
        period_df_time = period_df_time.drop(["label"], axis = 1)
    scaled_df = MinMaxScaler().fit_transform(period_df_time)
    k_number = find_optimal_k(scaled_df, is_test)
    label_result = k_mean_function(scaled_df, k_number)
    if is_test:
#         X_tsne = TSNE(n_components=2, random_state=123).fit_transform(scaled_df)
        _plot_kmean_scatter(period_df_time.values, label_result)
    return label_result


# In[23]:


def get_label_with_DBSAN(period_df, is_test = False):
    period_df_time = period_df.drop(["period", "weekday", "temperature", "segment_id", "isHot", "weather", "base_LOS", "district", "is_morning", "date"], axis=1)
    if "label" in period_df_time.columns:
        period_df_time = period_df_time.drop(["label"], axis = 1)
    scaled_df = MinMaxScaler().fit_transform(period_df_time)
    a = DBSAN_function(period_df_time)
    label_result = test_dbscan(scaled_df)
    if is_test:
        _plot_kmean_scatter(scaled_df, label_result)
    return label_result


parammeters = {
    "period": ["period_16_55"],
    "weekday": [1],
    "district": ["quan_binh_tan", "quan_10", "quan_tan_phu"],
#     "is_morning": [1]
}

def MinMaxScale_function(period_df):
    period_df_time = period_df.drop(["period", "weekday", "temperature", "segment_id", "isHot", "weather", "base_LOS", "district", "is_morning", "date"], axis=1, errors = 'raise')
    if "label" in period_df_time.columns:
        period_df_time = period_df_time.drop(["label"], axis = 1)
    scaled_df = MinMaxScaler().fit_transform(period_df_time)
    return scaled_df

def get_period_df(parameters, total_df): 
    period_df = total_df.copy()
    for key, value in parammeters.items():
        if value is not None:
            period_df = period_df.loc[period_df[key].isin(value)]
    return period_df

def get_results(parameters, total_df):
    period_df = get_period_df(parameters, total_df)
    kmeans_labels = get_label(period_df)
    period_df["label"] = kmeans_labels
    return period_df

if __name__ == "__main__":
    total_df = processing_data()
    print(get_results(parammeters, total_df).info())








