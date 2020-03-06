from .utils import read_jsonlines_lzma
from .tagger import Tagger
from .formatter import tweet_format
from .dataframe_loader import DataframeLoader
from . import (word2vec)

CATEGORIES = ["NEGATIVE", "NEUTRAL", "POSITIVE", "UNDEFINED"]

TAGS = {
    "POSITIVE": [
        ":-)",
        ":)",
        ":D",
        "=)",
    ],
    "NEGATIVE": [
        ":-(",
        ":(",
        "=(",
    ],
    "EMOJI_POSITIVE": [
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
    ],
    "EMOJI_NEGATIVE": [
        "😭",
        "😒",
        "😢",
        "😔",
        "😪",
    ]
}
