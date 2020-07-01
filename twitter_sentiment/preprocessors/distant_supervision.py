from typing import Iterable, Tuple
import json
import lzma
from tqdm import tqdm
import numpy as np
from twitter_sentiment.preprocessors import (POSITIVE_TOKENS, NEGATIVE_TOKENS,
                                             POSITIVE_CLASS, NEGATIVE_CLASS)
from twitter_sentiment.preprocessors.utils import read_jsonlines_lzma
from twitter_sentiment.preprocessors.dataset_preprocess import filter_lang
from twitter_sentiment.preprocessors.tokenizer import tokenize

def tag_sentiment(tweets: Iterable[dict], drop_multi_class: bool = True) -> Iterable[dict]:
    for tweet in tweets:
        tokens = tweet["tokenized_treated_text"]
        tags = set()

        for token in tokens:
            if token in POSITIVE_TOKENS:
                tags.add(POSITIVE_CLASS)
            elif token in NEGATIVE_TOKENS:
                tags.add(NEGATIVE_CLASS)

        if not tags:
            continue

        tags = list(tags)
        if drop_multi_class:
            if len(tags) > 1:
                continue
            tweet["distant_supervision_tags"] = tags[0]
        else:
            tweet["distant_supervision_tags"] = list(tags)

        if len(tags) == 1:
            tweet["distant_supervision_y"] = int(tags[0] == POSITIVE_CLASS)

        # remove tokens used in distant supervision
        tokens = [token for token in tokens if token not in (POSITIVE_TOKENS + NEGATIVE_TOKENS)]
        tweet["tokenized_treated_text"] = tokens

        if len(tokens) == 0:
            continue

        yield tweet

def distant_supervision_dataset(filepath: str, lang: str = 'pt') -> Iterable[dict]:
    """
    Get text from tweets given filepath.
    Uses original tweets (no retweets or quotes), filtered by language if available.
    Removes duplications.

    usage:
    >>> filepath = '/var/data/tweets.xz'
    >>> lang = 'pt'
    >>> tweets = dataset_preprocess(filepath, lang)

    :param filepath: path to lzma compressed tweets
    :param lang: language to filter, tweets without language field will be ignored. Default: 'pt'.
    :returns: generator of tweets
    """

    tweets = read_jsonlines_lzma(filepath)
    if lang:
        tweets = filter_lang(tweets, lang)
        tweets = tokenize(tweets, lang)
    else:
        tweets = tokenize(tweets)

    tweets = tag_sentiment(tweets)
    yield from tweets

def extract_tokenized_text_and_Y(tweets: Iterable[dict]) -> Tuple[list, np.ndarray]:
    X = list()
    Y = list()

    for t in tweets:
        if t.get("distant_supervision_y") is None:
            continue

        x = t["tokenized_treated_text"]
        y = t["distant_supervision_y"]
        X.append(x)
        Y.append(y)

    Y = np.array(Y)
    return X, Y

def _serial_distant_supervision(all_filepaths: Iterable[str], output_filepath: str, lang: str):

    with lzma.LZMAFile(output_filepath, mode="wb", format=lzma.FORMAT_XZ) as fout:
        for filename in all_filepaths:
            for tweet in tqdm(distant_supervision_dataset(filename, lang)):
                fout.write(json.dumps(tweet, separators=(',', ':')).encode('utf-8'))
                fout.write("\n".encode("utf-8"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs='+', help="Path to dataset", type=str)
    parser.add_argument("-o", "--output", help="Path to output file", type=str, required=True)
    parser.add_argument("-l", "--lang", help="Language", type=str, default="pt")
    args = parser.parse_args()
    _serial_distant_supervision(args.input, args.output, args.lang)
