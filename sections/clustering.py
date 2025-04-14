import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px


def clustering_section():
    st.header("🧩 Cluster Analysis Tool")
    st.markdown("Analyze and visualize clusters in your data using KMeans clustering.")

    uploaded_file = st.file_uploader("📂 Upload your CSV dataset", type=["csv"], key="clustering")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data")
        st.dataframe(df.head())

        all_columns = df.columns.tolist()
        selected_features = st.multiselect("Select numeric features for clustering", all_columns)

        if len(selected_features) < 2:
            st.info("Please select at least two numeric features.")
            return

        if st.checkbox("Remove rows with missing values", value=True):
            df = df.dropna(subset=selected_features)

        # Ensure selected features are numeric
        if not all(np.issubdtype(df[col].dtype, np.number) for col in selected_features):
            st.error("Selected features must be numeric.")
            return

        X = df[selected_features]
        k = st.slider("Choose number of clusters", 2, 10, 3)

        # Apply KMeans
        kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
        df['Cluster'] = kmeans.fit_predict(X)

        st.success("Clustering complete.")

        # Display cluster centers
        st.subheader("Cluster Centers (Feature Averages)")
        try:
            st.dataframe(df.groupby('Cluster')[selected_features].mean().reset_index())
        except Exception as e:
            st.warning(f"Unable to calculate cluster centers: {e}")

        # Visualization section
        st.subheader("Cluster Visualization")
        if len(selected_features) == 2:
            fig = px.scatter(
                df, x=selected_features[0], y=selected_features[1], color=df['Cluster'].astype(str),
                title="2D Cluster Visualization"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif len(selected_features) >= 3:
            pca = PCA(n_components=3)
            comps = pca.fit_transform(X)
            df[['PC1', 'PC2', 'PC3']] = comps
            fig = px.scatter_3d(
                df, x='PC1', y='PC2', z='PC3', color=df['Cluster'].astype(str),
                title="3D PCA Visualization"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Download section
        st.subheader("Download Results")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Clustered Dataset", csv, "clustered_data.csv", "text/csv")
