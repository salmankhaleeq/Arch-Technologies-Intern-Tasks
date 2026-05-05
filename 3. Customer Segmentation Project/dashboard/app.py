import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.title("Customer Segmentation Dashboard")

# Upload file
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    features = st.multiselect("Select Features", df.columns)

    if len(features) >= 2:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[features])

        k = st.slider("Select Number of Clusters", 2, 10, 5)

        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(scaled_data)

        df["Cluster"] = labels

        st.subheader("Clustered Data")
        st.write(df.head())

        # Plot
        fig, ax = plt.subplots()
        ax.scatter(df[features[0]], df[features[1]], c=labels)
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])

        st.pyplot(fig)