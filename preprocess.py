import os
import re
import string
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk

class TextPreprocessor:
    """
    Text preprocessing utility for cleaning, lemmatizing, and preparing text.
    """
    def __init__(self, stopwords_file=None):
        """
        Initialize the text preprocessor.
        :param stopwords_file: Path to a custom stopwords file.
        """
        self.stopwords = self._load_stopwords(stopwords_file)
        self.lemmatizer = WordNetLemmatizer()
        nltk.download("stopwords")
        nltk.download("punkt")
        nltk.download("wordnet")

    def _load_stopwords(self, file):
        """
        Load stopwords from a file or use default English stopwords.
        :param file: Path to stopwords file.
        :return: Set of stopwords.
        """
        stopwords_set = set(stopwords.words("english"))
        if file and os.path.exists(file):
            custom_stopwords = pd.read_csv(file, header=None).squeeze().tolist()
            stopwords_set.update(map(str.lower, custom_stopwords))
        return stopwords_set

    def preprocess_text(self, text):
        """
        Preprocess text by cleaning, removing stopwords, and lemmatizing.
        :param text: Input raw text.
        :return: Preprocessed text.
        """
        text = self._clean_text(text)
        text = self._remove_short_words(text)
        text = self._remove_stopwords(text)
        text = self._lemmatize_text(text)
        return text

    def _clean_text(self, text):
        """
        Clean text by removing punctuation, numbers, and special characters.
        :param text: Input text.
        :return: Cleaned text.
        """
        text = text.lower()
        text = re.sub(r'\b\d+[a-zA-Z_-]+\b', ' ', text)  # Remove patterns like 2-week
        text = re.sub(r'\[.*?\]', ' ', text)  # Remove text in square brackets
        text = re.sub(f"[{re.escape(string.punctuation)}]", ' ', text)  # Remove punctuation
        text = re.sub(r'\b\d+\b', ' ', text)  # Remove standalone numbers
        text = re.sub(r'[_-]', ' ', text)  # Replace underscores and hyphens with space
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        return re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters

    def _remove_stopwords(self, text):
        """
        Remove stopwords from the text.
        :param text: Input text.
        :return: Text without stopwords.
        """
        tokens = text.split()
        return " ".join([word for word in tokens if word not in self.stopwords])

    def _lemmatize_text(self, text):
        """
        Lemmatize text to reduce words to their base form.
        :param text: Input text.
        :return: Lemmatized text.
        """
        tokens = word_tokenize(text)
        return " ".join([self.lemmatizer.lemmatize(token) for token in tokens])

    def _remove_short_words(self, text):
        """
        Remove words with one or two characters.
        :param text: Input text.
        :return: Text without short words.
        """
        tokens = word_tokenize(text)
        return " ".join([word for word in tokens if len(word) > 2])
