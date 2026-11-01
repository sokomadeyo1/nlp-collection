"""Collection TF-IDF methods."""

from sklearn.feature_extraction.text import TfidfVectorizer

MAX_FEATURES = 100  # tf-idf embedding size


class CollectionTfidf:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=MAX_FEATURES)

    def tfidf_fit(self):
        self.tfidf.fit(self.texts())
        return self.tfidf

    def tfidf_transform(self):
        return self.tfidf.transform(self.texts())

    def tfidf_fit_transform(self):
        return self.tfidf.fit_transform(self.texts())

    def tfidf_lemmas_fit(self):
        if not self.lemmas:
            raise ValueError("Collection.lemmatize() must be called first")
        sent_lemmas = [" ".join(sent) for sent in self.lemmas]
        self.tfidf.fit(sent_lemmas)
        return self.tfidf

    def tfidf_lemmas_transform(self):
        if not self.lemmas:
            raise ValueError("Collection.lemmatize() must be called first")
        sent_lemmas = [" ".join(sent) for sent in self.lemmas]
        return self.tfidf.transform(sent_lemmas)

    def tfidf_lemmas_fit_transform(self):
        if not self.lemmas:
            raise ValueError("Collection.lemmatize() must be called first")
        sent_lemmas = [" ".join(sent) for sent in self.lemmas]
        return self.tfidf.fit_transform(sent_lemmas)
