from .utils import read_jsonlines_lzma

SPECIAL_TOKENS = {
    "mention": "<MENTION>",
    "link": "<LINK>",
}

POSITIVE_CLASS = "POSITIVE"
NEGATIVE_CLASS = "NEGATIVE"
NEUTRAL_CLASS = "NEUTRAL"
UNDEFINED_CLASS = "UNDEFINED"
CATEGORIES = [POSITIVE_CLASS, NEGATIVE_CLASS, NEUTRAL_CLASS, UNDEFINED_CLASS]
CLASSIFIER_CLASSES = [POSITIVE_CLASS, NEGATIVE_CLASS]

POSITIVE_TOKENS = [
    "😂",
    "❤",
    "😍",
    "💚",
    "💙",
    "💛",
    "💕",
    "😘",
    "♥",
    "😊",
]

NEGATIVE_TOKENS = [
    "😭",
    "😒",
    "😢",
    "😔",
    "😪",
]

ALL_TOKENS = POSITIVE_TOKENS + NEGATIVE_TOKENS
