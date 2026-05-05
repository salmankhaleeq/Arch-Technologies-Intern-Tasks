from src.data_preprocessing import load_data, clean_data, scale_data
from src.clustering import train_kmeans
from src.evaluation import evaluate_model
from src.utils import save_model

# Load data
df = load_data("data/raw/mall_customers.csv")

# Clean
df = clean_data(df)

# Features
features = ["Annual Income (k$)", "Spending Score (1-100)"]

# Scale
scaled_data, scaler = scale_data(df, features)

# Train model
model, labels = train_kmeans(scaled_data, n_clusters=5)

# Evaluate
score = evaluate_model(scaled_data, labels)

# Save model
save_model(model, "models/kmeans_model.pkl")

# Add labels
df["Cluster"] = labels

print("Model trained successfully!")
print(f"Silhouette Score: {score}")