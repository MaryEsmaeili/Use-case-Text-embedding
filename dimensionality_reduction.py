import umap.umap_ as umap
from sklearn.manifold import TSNE

class DimensionalityReduction:
    @staticmethod
    def umap_visualization(data, n_neighbors=5, min_dist=0.01):
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        embedding = reducer.fit_transform(data)
        return embedding

    @staticmethod
    def tsne_visualization(data, perplexity=30):
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embedding = tsne.fit_transform(data)
        return embedding
