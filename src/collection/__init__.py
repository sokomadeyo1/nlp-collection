"""Working with collections."""

__all__ = ["Collection"]

import pandas as pd

from ._preprocess import Preprocess, prepare_title_body
from ._morph import Morph
from ._lex import Lex
from ._statistics import Statistics
from ._tfidf import CollectionTfidf


class Collection(Preprocess, Morph, Lex, Statistics, CollectionTfidf):
    """Work with text collection."""

    def __init__(self, data=None):
        Preprocess.__init__(self)
        Morph.__init__(self)
        Lex.__init__(self)
        CollectionTfidf.__init__(self)
        self.data = None
        if data is not None:
            df = pd.read_csv(data)
            self.data = prepare_title_body(df)

    def __repr__(self) -> str:
        return self.data.__repr__()

    def texts(self) -> list[str]:
        """Get collection documents as plaintext."""
        return list(self.data)

    def lex_pipeline_min(self):
        """Perform minimal lexical analysis pipeline."""
        return self.tokenize().lemmatize()

    def lex_pipeline(self):
        """Perform basic lexical analysis pipeline.

        Takes a while to run."""
        return self.tokenize().remove_stop_words().lemmatize().stem()

    def lex_pipeline_full(self):
        """Perform full lexical analysis pipeline.

        Takes a lot of time to run."""
        return (
            self.tokenize()
            .remove_stop_words()
            .lemmatize()
            .stem()
            .get_sents()
            .get_sents_words()
        )

    def morph_pipeline(self):
        """Perform morphological analysis pipeline."""

        return self.tokenize().remove_stop_words().morph()

    def morph_pipeline_full(self):
        """Perform full morphological analysis pipeline.

        Takes a lot of time to run"""
        return self.tokenize().remove_stop_words().lemmatize().morph().morph_unique()
