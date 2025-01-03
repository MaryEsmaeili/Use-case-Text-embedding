from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

class TopicModeling:
    @staticmethod
    def lda_modeling(data, n_topics=10, max_features=1000):
        vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
        data_vectorized = vectorizer.fit_transform(data)

        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=10)
        lda_topics = lda.fit_transform(data_vectorized)

        feature_names = vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(lda.components_):
            print(f"Topic {topic_idx + 1}:", ", ".join([feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]))
        return lda_topics

    @staticmethod
    def nmf_modeling(data, n_topics=10, max_features=1000):
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        data_vectorized = vectorizer.fit_transform(data)

        nmf = NMF(n_components=n_topics, random_state=42, max_iter=100)
        nmf_topics = nmf.fit_transform(data_vectorized)

        feature_names = vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(nmf.components_):
            print(f"Topic {topic_idx + 1}:", ", ".join([feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]))
        return nmf_topics
