from .utils import read_jsonlines_lzma
from .formatter import tweet_format

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
