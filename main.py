import os
import glob
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from topic_visualization import TopicVisualization
from preprocess import TextPreprocessor
from features import FeatureExtractor
from models import TopicModeling
from clustering import Clustering
from dimensionality_reduction import DimensionalityReduction
from nltk_setup import download_nltk_resources


def load_and_extract_files(zip_file_path, extraction_directory):
    """
    Load and extract files from a zip archive, then read .txt files into a DataFrame.
    """
    # Validate the zip file
    if not os.path.exists(zip_file_path):
        raise FileNotFoundError(f"The specified zip file {zip_file_path} does not exist.")

    # Extract the zip file if not already extracted
    if not os.path.exists(extraction_directory):
        os.makedirs(extraction_directory, exist_ok=True)
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extraction_directory)
        print(f"Extracted {zip_file_path} to {extraction_directory}.")

    # Get all .txt files in the extracted directory
    docs = glob.glob(os.path.join(extraction_directory, "*.txt"))

    # Check if documents are found
    if not docs:
        raise FileNotFoundError(f"No .txt files found in {extraction_directory}. Ensure the dataset is correctly extracted.")

    # Create a DataFrame to store the documents
    df = pd.DataFrame(columns=['docid', 'text'])

    # Read and populate the DataFrame
    for doc in docs:
        txt = Path(doc).read_text(encoding='utf-8')  # Ensure encoding handles special characters
        doc_id = Path(doc).stem  # Extract document ID (filename without extension)
        df.loc[len(df.index)] = [doc_id, txt]

    # Set the index to 'docid'
    return df.set_index('docid')


def main():
    # Ensure NLTK resources are available
    download_nltk_resources()

    # Define paths for zip file and extraction directory
    zip_file_path = r"D:\Data Science\Data Science 6\assignments\Use case Text embedding\data\MACCROBAT2020.zip"
    extraction_directory = r"MACCROBAT2020_extracted"

    # Load and preprocess data
    df = load_and_extract_files(zip_file_path, extraction_directory)

    # Preprocess text
    preprocessor = TextPreprocessor(stopwords_file="stopwords.csv")
    df["text"] = df["text"].apply(preprocessor.preprocess_text)
    data = df["text"].values

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
    data_tfidf = vectorizer.fit_transform(data)

    # Topic Modeling (NMF)
    nmf_model = NMF(n_components=5, random_state=42)
    nmf_model.fit(data_tfidf)

    # Display Topics
    feature_names = vectorizer.get_feature_names_out()
    TopicVisualization.display_topics(nmf_model, feature_names, num_top_words=5)

    # Plot Topics
    TopicVisualization.plot_topic_bars(nmf_model, feature_names, num_top_words=5)


    # Step 1: Feature Extraction
    print("\n=== Feature Extraction ===")
    tfidf, tf, tfidf_vectorizer, tf_vectorizer = FeatureExtractor.extract_features(df, "text")

    # Topic Modeling
    print("\n=== LDA Modeling ===")
    lda_topics = TopicModeling.lda_modeling(data, n_topics=5)

    print("\n=== NMF Modeling ===")
    nmf_topics = TopicModeling.nmf_modeling(data, n_topics=5)

    # Combine topic embeddings
    combined_topics = np.hstack((lda_topics, nmf_topics))

    # Dimensionality Reduction
    print("\n=== Dimensionality Reduction ===")
    lda_umap = DimensionalityReduction.umap_visualization(lda_topics)
    nmf_umap = DimensionalityReduction.umap_visualization(nmf_topics)
    combined_umap = DimensionalityReduction.umap_visualization(combined_topics)

    # Clustering and Visualization
    print("\n=== Clustering and Visualization ===")

    # 1. LDA with MiniBatchKMeans
    lda_labels, lda_inertia = Clustering.minibatch_kmeans_clustering(lda_umap)
    lda_silhouette, lda_davies_bouldin = Clustering.evaluate_clustering(lda_umap, lda_labels)
    Clustering.plot_clusters(lda_umap, lda_labels, title="LDA with MiniBatchKMeans")
    print(f"LDA MiniBatchKMeans - Silhouette Score: {lda_silhouette}, Davies-Bouldin Index: {lda_davies_bouldin}, Inertia: {lda_inertia}")

    # 2. NMF with MiniBatchKMeans
    nmf_labels, nmf_inertia = Clustering.minibatch_kmeans_clustering(nmf_umap)
    nmf_silhouette, nmf_davies_bouldin = Clustering.evaluate_clustering(nmf_umap, nmf_labels)
    Clustering.plot_clusters(nmf_umap, nmf_labels, title="NMF with MiniBatchKMeans")
    print(f"NMF MiniBatchKMeans - Silhouette Score: {nmf_silhouette}, Davies-Bouldin Index: {nmf_davies_bouldin}, Inertia: {nmf_inertia}")

    # 3. Combined LDA+NMF with MiniBatchKMeans
    combined_labels_mb, combined_inertia_mb = Clustering.minibatch_kmeans_clustering(combined_umap)
    combined_mb_silhouette, combined_mb_davies_bouldin = Clustering.evaluate_clustering(combined_umap, combined_labels_mb)
    Clustering.plot_clusters(combined_umap, combined_labels_mb, title="Combined LDA+NMF with MiniBatchKMeans")
    print(f"Combined LDA+NMF MiniBatchKMeans - Silhouette Score: {combined_mb_silhouette}, Davies-Bouldin Index: {combined_mb_davies_bouldin}, Inertia: {combined_inertia_mb}")

    # 4. Combined LDA+NMF with KMeans++
    combined_labels_kmeans, combined_inertia_kmeans = Clustering.kmeans_clustering(combined_umap)
    combined_kmeans_silhouette, combined_kmeans_davies_bouldin = Clustering.evaluate_clustering(combined_umap, combined_labels_kmeans)
    Clustering.plot_clusters(combined_umap, combined_labels_kmeans, title="Combined LDA+NMF with KMeans++")
    print(f"Combined LDA+NMF KMeans++ - Silhouette Score: {combined_kmeans_silhouette}, Davies-Bouldin Index: {combined_kmeans_davies_bouldin}, Inertia: {combined_inertia_kmeans}")

if __name__ == "__main__":
    main()