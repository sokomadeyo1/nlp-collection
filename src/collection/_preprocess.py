"""Primary text preprocessing."""

from re import sub
from natasha import Doc, Segmenter
from nltk.corpus import stopwords


segmenter = Segmenter()
STOP = stopwords.words("russian")  # + ["приятного", "аппетита"]
STOP = STOP + [".", ",", "!", "?", "(", ")", '"', "'", ":"]


def prepare_title_body(df):
    """Concatenate title and body."""

    def sentend(sent):
        if sent[-1].isalpha():
            return sent + "."
        return sent

    return df["title"].apply(sentend) + "\n" + df["body"]


class Preprocess:
    """Collection preprocessing methods."""

    def __init__(self):
        self.docs = None  # Natasha
        self.tokens = None

    def clean(self):
        """Remove unnecessary tokens such as page numbers.
        Must be performed before tokenization.

        Returns:
            self -- Collection of clean texts.
        """

        def remove_numbers(s):
            pattern = r"\n\s*\d+\s*\n"  # lines with just one number
            return sub(pattern, "", s)

        # def some_other_preprocess_step(s):
        # ...

        self.data = self.data.apply(remove_numbers).apply(str.strip)

        return self

    # TODO: Rewrite to make it usable only after tokenization
    def remove_stop_words(self, stop=STOP):
        """Remove stop words from all documents of collection.

        Args:
            stop: list[str] -- List of stop words.

        Returns:
            self -- Collection without stop words.
        """

        if not self.tokens:
            raise ValueError("Collection.tokenize() must be called first")

        self.tokens = [[w for w in doc if w.lower() not in stop] for doc in self.tokens]
        self.docs = [[w for w in doc.tokens if w.text.lower() not in stop] for doc in self.docs]

        return self

    def tokenize(self):
        """Tokenize collection."""
        self.docs = [Doc(doc) for doc in self.data]
        for doc in self.docs:
            doc.segment(segmenter)

        self.tokens = [[token.text for token in doc.tokens] for doc in self.docs]
        return self
