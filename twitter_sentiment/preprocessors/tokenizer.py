import string
from typing import Iterable
import nltk
from nltk.corpus import stopwords


def tokenize(tweets: Iterable[dict], lang: str = None, reduce_len: bool = True,
             remove_single: bool = False, remove_stopwords: bool = False,
             remove_punctuation: bool = True) -> Iterable[dict]:

    if remove_stopwords and (not lang):
        raise RuntimeError("Missing language parameter")

    if lang:
        lang_map = {"pt": "portuguese", "en": "english"}
        stop = set(stopwords.words(lang_map[lang]))

    tokenizer = nltk.tokenize.casual.TweetTokenizer(reduce_len=reduce_len)

    for tweet in tweets:
        text = tweet["treated_text"]
        tokens = tokenizer.tokenize(text)

        # removes tweets with single word
        if remove_single:
            tokens = [t for t in tokens if len(t) >= 2]

        if remove_stopwords:
            tokens = [t for t in tokens if (t not in stop)]

        if remove_punctuation:
            tokens = [t for t in tokens if (t not in string.punctuation)]

        tweet["tokenized_treated_text"] = tokens
        # TODO maybe remove punctuation tokens?
        yield tweet
