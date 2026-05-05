from sklearn.metrics import silhouette_score

def evaluate_model(data, labels):
    score = silhouette_score(data, labels)
    return score