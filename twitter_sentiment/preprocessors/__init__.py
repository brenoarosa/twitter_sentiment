from .utils import read_jsonlines_lzma

SPECIAL_TOKENS = {
    "mention": "<MENTION>",
    "link": "<LINK>",
}

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
