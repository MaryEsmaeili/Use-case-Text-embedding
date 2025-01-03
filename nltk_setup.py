import nltk

def download_nltk_resources():

    """
    Ensure that necessary NLTK resources are downloaded and available.
    Avoids repeated download checks by verifying their presence.
    """
    resources = ["punkt", "wordnet", "stopwords"]
    for resource in resources:
        try:
            # Check if the resource is already downloaded
            nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
        except LookupError:
            # Download resource if not found
            nltk.download(resource)