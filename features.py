from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

class FeatureExtractor:
    @staticmethod
    def extract_features(df, column, n_features=1000):
        """
        Extract TF-IDF and CountVectorizer features from text.
        """
        tfidf_vectorizer = TfidfVectorizer(
            max_df=0.8, min_df=0.01, max_features=n_features, stop_words="english"
        )
        tfidf = tfidf_vectorizer.fit_transform(df[column])

        tf_vectorizer = CountVectorizer(
            max_df=0.8, min_df=0.01, max_features=n_features, stop_words="english"
        )
        tf = tf_vectorizer.fit_transform(df[column])

        return tfidf, tf, tfidf_vectorizer, tf_vectorizer