import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna()
    return df

def scale_data(df, features):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[features])
    return scaled, scaler