from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

class Clustering:
    @staticmethod
    def minibatch_kmeans_clustering(data, n_clusters=10):
        """
        Perform clustering using MiniBatchKMeans.
        :param data: Input data (e.g., UMAP or t-SNE embeddings).
        :param n_clusters: Number of clusters.
        :return: Cluster labels and inertia.
        """
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=100)
        labels = kmeans.fit_predict(data)
        return labels, kmeans.inertia_

    @staticmethod
    def kmeans_clustering(data, n_clusters=10):
        """
        Perform clustering using KMeans++.
        :param data: Input data (e.g., UMAP or t-SNE embeddings).
        :param n_clusters: Number of clusters.
        :return: Cluster labels and inertia.
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, init="k-means++", n_init=10)
        labels = kmeans.fit_predict(data)
        return labels, kmeans.inertia_

    @staticmethod
    def evaluate_clustering(data, labels):
        """
        Evaluate clustering using silhouette and Davies-Bouldin scores.
        :param data: Input data (e.g., UMAP or t-SNE embeddings).
        :param labels: Cluster labels.
        :return: Silhouette and Davies-Bouldin scores.
        """
        if len(set(labels)) > 1:
            silhouette = silhouette_score(data, labels)
            davies_bouldin = davies_bouldin_score(data, labels)
            return silhouette, davies_bouldin
        else:
            return None, None

    @staticmethod
    def plot_clusters(embedding, labels, title="Clustering Visualization"):
        """
        Plot the clustering results on a 2D embedding.
        :param embedding: 2D array-like data (e.g., UMAP or t-SNE output).
        :param labels: Cluster labels for the data points.
        :param title: Title of the plot.
        """
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            embedding[:, 0], embedding[:, 1], c=labels, cmap="viridis", s=50, alpha=0.8
        )
        plt.colorbar(scatter, label="Cluster")
        plt.title(title)
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(alpha=0.3)
        plt.show()
