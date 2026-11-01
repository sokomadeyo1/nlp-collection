"""Statistics calculation."""

from nltk.probability import FreqDist
from .util import flatten, plot_dist

UPOS_TAGS = [
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
]


class Statistics:
    """Collection statistics methods."""

    def word_freq(self):
        """Get collection words distribution."""
        return FreqDist(token.lower() for token in flatten(self.tokens))

    def word_freq_per_document(self):
        """Get collections documents words distribution."""
        return [FreqDist(doc) for doc in self.tokens]

    def word_lens(self):
        """Get collection words lengths distribution."""
        if not self.tokens:
            raise ValueError("Collection.tokenize() must be called first")
        dist = FreqDist()
        for token in flatten(self.tokens):
            dist[len(token)] += 1
        return dist

    def mean_word_len(self):
        """Get collections mean word length."""
        dist = self.word_lens()
        m = 0
        for len in dist.keys():
            m += len * dist.freq(len)
        return m

    def lexical_density(self):
        """Get collections lexical density."""
        if not self.lemmas:
            raise ValueError("Collection.lemmatize() must be called first")

        def lemmas_count():
            unique_lemmas = set()
            for w in flatten(self.lemmas):
                unique_lemmas.add(w)
            return len(unique_lemmas)

        def wordforms_count():
            unique_words = set()
            for w in flatten(self.tokens):
                unique_words.add(w)
            return len(unique_words)

        return lemmas_count() / wordforms_count()

    def word_freq_per_pos(self):
        """Get collections pos words distributions."""
        if not self.lemmas:
            raise ValueError("Collection.lemmatize() must be called first")
        freqs = dict.fromkeys(UPOS_TAGS)
        for pos in freqs.keys():
            tokens = flatten([doc.tokens for doc in self.docs])
            words = [token.text for token in tokens if token.pos == pos]
            freqs[pos] = FreqDist(words)

        return freqs

    def pos_words_count(self):
        """Get number of different words for each pos."""
        if not self.lemmas:
            raise ValueError("Collection.lemmatize() must be called first")
        pos_dict = dict.fromkeys(UPOS_TAGS)
        for pos in pos_dict.keys():
            tokens = flatten([doc.tokens for doc in self.docs])
            words = set([token.text for token in tokens if token.pos == pos])
            pos_dict[pos] = len(words)
        return pos_dict

    def pos_lemmas_count(self):
        """Get number of different lemmas for each pos."""
        if not self.lemmas:
            raise ValueError("Collection.lemmatize() must be called first")
        pos_dict = dict.fromkeys(UPOS_TAGS)
        for pos in pos_dict.keys():
            tokens = flatten([doc.tokens for doc in self.docs])
            lemmas = set([token.lemma for token in tokens if token.pos == pos])
            pos_dict[pos] = len(lemmas)
        return pos_dict

    def pos_unambiguous_word_count(self) -> int:
        """Count words with no POS ambiguity."""

        def POSs(parse):
            return set([p.tag.POS for p in parse])

        unambiguous = [
            parse.text for parse in flatten(self.parse_unique) if len(POSs(parse)) == 1
        ]
        return len(set(unambiguous))

    def homonyms_count(self) -> int:
        """Get homonyms count."""
        return len(set(self.homonyms_unique()))

    def homonyms_dist(self):
        """Distribution of homonyms by type."""
        if not self.parse:
            raise ValueError("Collection.morph() must be called first")

        def POS_homonym(parse) -> bool:
            return len(set([p.tag.POS for p in parse])) > 1

        dist = FreqDist(
            {
                "POS": len(
                    [p[0].word for p in flatten(self.parse) if POS_homonym(p)]
                ),
                "other": len(
                    [p[0].word for p in flatten(self.parse) if not POS_homonym(p)]
                ),
            }
        )

        return dist

    def words_by_cases_dist(self):
        """Distribution of collection words by cases."""
        dict = self.words_by_cases()
        dist = {key: len(dict[key]) for key in dict.keys()}
        return FreqDist(dist)

    def words_by_genders_dist(self):
        """Distribution of collection words by genders."""
        dict = self.words_by_genders()
        dist = {key: len(dict[key]) for key in dict.keys()}
        return FreqDist(dist)

    def words_by_numbers_dist(self):
        """Distribution of collection words by numbers."""
        dict = self.words_by_numbers()
        dist = {key: len(dict[key]) for key in dict.keys()}
        return FreqDist(dist)

    def nouns_by_animacy_dist(self):
        """Distribution of collection nouns by animacy."""
        dict = self.nouns_by_animacy()
        dist = {key: len(dict[key]) for key in dict.keys()}
        return FreqDist(dist)

    def verbs_by_conjugations_dist(self):
        """Distribution of collection verbs by conjugations."""
        dict = self.verbs_by_conjugations()
        dist = {key: len(dict[key]) for key in dict.keys()}
        return FreqDist(dist)

    def verbs_by_aspects_dist(self):
        """Distribution of collection verbs by aspect."""
        dict = self.verbs_by_aspects()
        dist = {key: len(dict[key]) for key in dict.keys()}
        return FreqDist(dist)

    def verbs_by_tenses_dist(self):
        """Distribution of collection verbs by tenses."""
        dict = self.verbs_by_tenses()
        dist = {key: len(dict[key]) for key in dict.keys()}
        return FreqDist(dist)

    # Distribution plotting methods

    def plot_frequent_words(self, N=20):
        """Draw a barplot of collections most frequent words distribution."""
        plot_dist(self.word_freq().most_common(N))

    def plot_lens(self):
        """Draw a barplot of collections word lengths distribution."""
        plot_dist(self.word_lens())

    def plot_pos_words(self):
        """Draw a barplot of words distribution by pos."""
        plot_dist(self.pos_words_count())

    def plot_pos_lemmas(self):
        """Draw a barplot of words distribution by pos."""
        plot_dist(self.pos_lemmas_count())

    def plot_homonym_types(self):
        """Draw a barplot of homonyms distribution by type."""
        plot_dist(self.homonyms_dist())

    def plot_words_by_cases_dist(self):
        """Draw a barplot of distribution of words by cases."""
        plot_dist(self.words_by_cases_dist())

    def plot_words_by_genders_dist(self):
        """Draw a barplot of distribution of words by genders."""
        plot_dist(self.words_by_genders_dist())

    def plot_words_by_numbers_dist(self):
        """Draw a barplot of distribution of words by numbers."""
        plot_dist(self.words_by_numbers_dist())

    def plot_nouns_by_animacy_dist(self):
        """Draw a barplot of distribution of nouns by animacy."""
        plot_dist(self.nouns_by_animacy_dist())

    def plot_verbs_by_conjugations_dist(self):
        """Draw a barplot of distribution of verbs by conjugations."""
        plot_dist(self.verbs_by_conjugations_dist())

    def plot_verbs_by_aspects_dist(self):
        """Draw a barplot of distribution of verbs by aspect."""
        plot_dist(self.verbs_by_aspects_dist())

    def plot_verbs_by_tenses_dist(self):
        """Draw a barplot of distribution of verbs by tenses."""
        plot_dist(self.verbs_by_tenses_dist())
