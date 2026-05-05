from sklearn.cluster import KMeans

def train_kmeans(data, n_clusters=5):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(data)
    return model, labels