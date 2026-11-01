"""Morphological analysis."""

from pymorphy2 import MorphAnalyzer
from re import match
from .util import flatten

morph = MorphAnalyzer()

# Measures against hardcoding
NOUN = "NOUN"
VERB = "VERB"

FIRST_CONJ_EXCEPTIONS = ["брить", "зиждиться", "зыбиться", "стелить"]
SECOND_CONJ_EXCEPTIONS = [
    "смотреть",
    "вертеть",
    "видеть",
    "зависеть",
    "ненавидеть",
    "обидеть",
    "терпеть",
    "гнать",
    "дышать",
    "держать",
    "слышать",
]

FIRST_CONJ_SUFFIXES = ["ать", "ять", "еть", "уть", "ють", "оть"]
SECOND_CONJ_SUFFIXES = ["ить"]

ANIMACY = frozenset(
    [
        "anim",  # одушевлённое
        "inan",  # неодушевлённое
    ]
)

GENDERS = frozenset(
    [
        "masc",  # мужской род
        "femn",  # женский род
        "neut",  # средний род
    ]
)

NUMBERS = frozenset(
    [
        "sing",  # единственное число
        "plur",  # множественное число
    ]
)

CASES = frozenset(
    [
        "nomn",  # именительный падеж
        "gent",  # родительный падеж
        "datv",  # дательный падеж
        "accs",  # винительный падеж
        "ablt",  # творительный падеж
        "loct",  # предложный падеж
        "voct",  # звательный падеж
        "gen1",  # первый родительный падеж
        "gen2",  # второй родительный (частичный) падеж
        "acc2",  # второй винительный падеж
        "loc1",  # первый предложный падеж
        "loc2",  # второй предложный (местный) падеж
    ]
)

ASPECTS = frozenset(
    [
        "perf",  # совершенный вид
        "impf",  # несовершенный вид
    ]
)

TENSES = frozenset(
    [
        "pres",  # настоящее время
        "past",  # прошедшее время
        "futr",  # будущее время
    ]
)

CONJUGATIONS = frozenset(
    [
        "1",
        "2",
        "other",
    ]
)


def verb_conj(verb_parse):
    if verb_parse.tag.POS != VERB:
        raise ValueError(f"{verb_parse} is not a verb")

    w = verb_parse.word
    norm = verb_parse.normal_form
    if w in FIRST_CONJ_EXCEPTIONS:
        return "1"
    elif w in SECOND_CONJ_EXCEPTIONS:
        return "2"
    elif any([match(r".*" + suffix + r".*", norm) for suffix in FIRST_CONJ_SUFFIXES]):
        return "1"
    elif any([match(r".*" + suffix + r".*", norm) for suffix in SECOND_CONJ_SUFFIXES]):
        return "2"
    return "other"


class Morph:
    """Collection morph methods."""

    def __init__(self):
        self.parse = None
        self.parse_unique = None

    def morph(self):
        if not self.tokens:
            raise ValueError("Collection.tokenize() must be called first")

        self.parse = [[morph.parse(token) for token in doc] for doc in self.tokens]
        return self

    def morph_unique(self):
        if not self.tokens:
            raise ValueError("Collection.tokenize() must be called first")

        self.parse_unique = [morph.parse(token) for token in set(flatten(self.tokens))]
        return self

    def unambiguous(self):
        """Get all unambiguous words."""
        if not self.parse:
            raise ValueError("Collection.morph() must be called first")

        return [p[0].text for p in flatten(self.parse) if len(p) == 1]

    def homonyms(self) -> list[str]:
        """Get all homonyms."""
        if not self.parse:
            raise ValueError("Collection.morph() must be called first")

        return [p[0].text for p in flatten(self.parse) if len(p) > 1]

    def unambiguous_unique(self):
        """Get all unambiguous words."""
        if not self.parse:
            raise ValueError("Collection.morph_unique() must be called first")

        return [p[0].text for p in self.parse_unique if len(p) == 1]

    def homonyms_unique(self) -> list[str]:
        """Get all homonyms."""
        if not self.parse:
            raise ValueError("Collection.morph_unique() must be called first")

        return [p[0].text for p in self.parse_unique if len(p) > 1]

    def words_by_cases(self):
        """Dict of words by cases."""
        if not self.parse:
            raise ValueError("Collection.morph() must be called first")

        return {
            case: [p for p in flatten(self.parse) if p[0].tag.case == case]
            for case in CASES
        }

    def words_by_genders(self):
        """Dict of words by genders."""
        if not self.parse:
            raise ValueError("Collection.morph() must be called first")

        return {
            gender: [p for p in flatten(self.parse) if p[0].tag.gender == gender]
            for gender in GENDERS
        }

    def words_by_numbers(self):
        """Dict of words by numbers."""
        if not self.parse:
            raise ValueError("Collection.morph() must be called first")

        return {
            number: [p for p in flatten(self.parse) if p[0].tag.number == number]
            for number in NUMBERS
        }

    def nouns_by_animacy(self):
        """Dict of nouns by animacy."""
        if not self.parse:
            raise ValueError("Collection.morph() must be called first")

        nouns = [p for p in flatten(self.parse) if p[0].tag.POS == NOUN]
        return {
            animacy: [p for p in nouns if p[0].tag.animacy == animacy]
            for animacy in ANIMACY
        }

    def verbs_by_conjugations(self):
        """Dict of verbs by conjugations."""
        if not self.parse:
            raise ValueError("Collection.morph() must be called first")

        verbs = [p for p in flatten(self.parse) if p[0].tag.POS == VERB]
        return {
            conjugation: [p for p in verbs if verb_conj(p[0]) == conjugation]
            for conjugation in CONJUGATIONS
        }

    def verbs_by_aspects(self):
        """Dict of verbs by aspects."""
        if not self.parse:
            raise ValueError("Collection.morph() must be called first")

        verbs = [p for p in flatten(self.parse) if p[0].tag.POS == VERB]
        return {
            aspect: [p for p in verbs if p[0].tag.aspect == aspect]
            for aspect in ASPECTS
        }

    def verbs_by_tenses(self):
        """Dict of verbs by tenses."""
        if not self.parse:
            raise ValueError("Collection.morph() must be called first")

        verbs = [p for p in flatten(self.parse) if p[0].tag.POS == VERB]
        return {
            tense: [p for p in verbs if p[0].tag.tense == tense]
            for tense in TENSES
        }
