"""Lexical analysis."""

from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from natasha import (
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
)
from .util import flatten

morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
stemmer = SnowballStemmer("russian")


def sent_wc(sent):
    return len(word_tokenize(sent))


class Lex:
    """Collection lexical analysis methods."""

    def __init__(self):
        self.lemmas = None
        self.stems = None
        self.sents = None
        self.sents_words = None

    def lemmatize(self):
        """Lemmatize collection."""
        if not self.docs:
            raise ValueError("Collection.tokenize() must be called first")
        for doc in self.docs:
            doc.tag_morph(morph_tagger)
        # flatten self.docs
        for token in [t for doc in self.docs for t in doc.tokens]:
            token.lemmatize(morph_vocab)
        self.lemmas = [[token.lemma for token in doc.tokens] for doc in self.docs]
        return self

    def stem(self):
        """Get collection stems."""
        if not self.docs:
            raise ValueError("Collection.tokenize() must be called first")
        self.stems = [[stemmer.stem(token) for token in doc] for doc in self.tokens]
        return self

    def words_all(self):
        """Flat list of all words for morphological analysis."""
        if not self.docs:
            raise ValueError("Collection.tokenize() must be called first")
        return flatten(self.docs)

    def get_sents(self):
        """Save sentences texts to self.sents."""
        if not self.docs:
            raise ValueError("Collection.tokenize() must be called first")
        self.sents = [[sent.text for sent in doc.sents] for doc in self.docs]
        return self

    def get_sents_words(self):
        """Save sentences natasha tokens to self.sents_words."""
        if not self.docs:
            raise ValueError("Collection.tokenize() must be called first")
        self.sents_words = [[sent.tokens for sent in doc.sents] for doc in self.docs]
        return self

    def avg_sent_len(self):
        """Calculate collections average sentence length.

        Length is measured in tokens, dropping stop words and/or punctuation is up to the user."""
        if not self.sents:
            raise ValueError("Collection.get_sents() must be called first")

        def avg(list):
            return sum(list) / len(list)

        return avg([sent_wc(sent) for sent in flatten(self.sents)])

    # число предложений в k-m тексте = len(self.sents[k])
    # число предложений в коллекции = len(flatten(self.sents))
    # среднее число предложений в коллекции = len(flatten(self.sents)) / len(self.data)
