from  sklearn.cluster import DBSCAN, KMeans

def cluster_dbscan(features, eps=1.75, min_samples=2, metric='euclidean'):
    return DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(features).labels_

def cluster_kmeans(features, n_clusters=3):
    return KMeans(n_clusters=n_clusters).fit(features).labels_