from typing import Generator
import nltk
from nltk.corpus import stopwords


def tokenize(tweets: dict, lang: str = None, reduce_len: bool = True,
             remove_single: bool = False, remove_stopwords: bool = False) -> Generator[dict, None, None]:

    if remove_stopwords and (not lang):
        raise RuntimeError("Missing language parameter")

    if lang:
        lang_map = {"pt": "portuguese", "en": "english"}
        tokenizer = nltk.tokenize.casual.TweetTokenizer(reduce_len=reduce_len)
        stop = set(stopwords.words(lang_map[lang]))

    for tweet in tweets:
        text = tweet["treated_text"]
        tokens = tokenizer.tokenize(text)

        # removes tweets with single word
        if remove_single:
            tokens = filter(lambda x: len(x) >= 2, tokens)

        if remove_stopwords:
            tokens = filter(lambda x: x not in stop, tokens)

        tweet["tokenized_treated_text"] = tokens
        # TODO maybe remove punctuation tokens?
        yield tweet
