import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from openTSNE import TSNE
from umap import UMAP
import plotly.express as px

# Title and description
st.title("Customer Segmentation")
st.write("""
## Using Unsupervised Learning Techniques for Customer Segmentation
This app demonstrates various clustering and dimensionality reduction techniques on an e-commerce dataset.
""")

# Load data
@st.cache
def load_data():
    url = "https://drive.google.com/uc?export=download&id=1lEccW5Y5_2z00VRtLGOAJOAU6YA9fl6W"
    df = pd.read_csv(url)
    return df

df = load_data()
st.write("### Dataset Head", df.head())

# Preprocessing
scaler = MinMaxScaler()
x = df.drop('ID', axis=1)
x_scaled = scaler.fit_transform(x)

# Sidebar for user input
st.sidebar.header("Choose Clustering Algorithm")
algorithm = st.sidebar.selectbox("Algorithm", ["K-Means", "GMM", "Hierarchical", "DBSCAN"])

if algorithm == "K-Means":
    k = st.sidebar.slider("Number of clusters (K)", 2, 10, 4)
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_pred = kmeans.fit_predict(x_scaled)
    df['Cluster'] = y_pred
    st.write("### Cluster Centers", kmeans.cluster_centers_)
    
    # Visualization
    fig, ax = plt.subplots()
    sns.scatterplot(x=x_scaled[:, 0], y=x_scaled[:, 1], hue=y_pred, palette="viridis", ax=ax)
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='red', marker='X', s=100)
    st.pyplot(fig)

elif algorithm == "GMM":
    k = st.sidebar.slider("Number of clusters (K)", 2, 10, 4)
    gmm = GaussianMixture(n_components=k, random_state=42)
    y_pred = gmm.fit_predict(x_scaled)
    df['Cluster'] = y_pred
    st.write("### GMM Means", gmm.means_)
    
    # Visualization
    fig, ax = plt.subplots()
    sns.scatterplot(x=x_scaled[:, 0], y=x_scaled[:, 1], hue=y_pred, palette="viridis", ax=ax)
    st.pyplot(fig)

elif algorithm == "Hierarchical":
    k = st.sidebar.slider("Number of clusters (K)", 2, 10, 4)
    hc = AgglomerativeClustering(n_clusters=k)
    y_pred = hc.fit_predict(x_scaled)
    df['Cluster'] = y_pred
    
    # Visualization
    fig, ax = plt.subplots()
    sns.scatterplot(x=x_scaled[:, 0], y=x_scaled[:, 1], hue=y_pred, palette="viridis", ax=ax)
    st.pyplot(fig)

elif algorithm == "DBSCAN":
    eps = st.sidebar.slider("Epsilon (eps)", 0.1, 10.0, 0.5)
    min_samples = st.sidebar.slider("Minimum Samples", 1, 10, 5)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    y_pred = dbscan.fit_predict(x_scaled)
    df['Cluster'] = y_pred
    
    # Visualization
    fig, ax = plt.subplots()
    sns.scatterplot(x=x_scaled[:, 0], y=x_scaled[:, 1], hue=y_pred, palette="viridis", ax=ax)
    st.pyplot(fig)

# Dimensionality Reduction
st.sidebar.header("Dimensionality Reduction")
dr_method = st.sidebar.selectbox("Method", ["PCA", "t-SNE", "UMAP"])

if dr_method == "PCA":
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_scaled)
    st.write("### Explained Variance Ratio", pca.explained_variance_ratio_)
    
    fig, ax = plt.subplots()
    sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1], hue=df['Cluster'], palette="viridis", ax=ax)
    st.pyplot(fig)

elif dr_method == "t-SNE":
    tsne = TSNE(n_jobs=-1, random_state=42)
    x_tsne = tsne.fit(x_scaled)
    
    fig, ax = plt.subplots()
    sns.scatterplot(x=x_tsne[:, 0], y=x_tsne[:, 1], hue=df['Cluster'], palette="viridis", ax=ax)
    st.pyplot(fig)

elif dr_method == "UMAP":
    umap = UMAP(random_state=42)
    x_umap = umap.fit_transform(x_scaled)
    
    fig, ax = plt.subplots()
    sns.scatterplot(x=x_umap[:, 0], y=x_umap[:, 1], hue=df['Cluster'], palette="viridis", ax=ax)
    st.pyplot(fig)
