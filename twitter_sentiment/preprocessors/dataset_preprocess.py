from typing import Generator
import json
import lzma
import re
import unicodedata
import joblib
from tqdm import tqdm
from twitter_sentiment.preprocessors import SPECIAL_TOKENS
from twitter_sentiment.preprocessors.utils import read_jsonlines_lzma

def get_original_posts(tweets: dict) -> Generator[dict, None, None]:
    for tweet in tweets:
        user_field = "user_info"
        text_field = "text"
        id_field = "id"

        # use quoted tweets same as original
        if "retweeted_status" in tweet.keys():
            tweet = tweet["retweeted_status"]
            user_field = "user"
            text_field = "full_text"
            id_field = "id_str"

        simple_tweet = {
            "user_id": tweet[user_field]["id_str"],
            "text": tweet[text_field],
            "id": tweet[id_field],
            "lang": tweet["lang"],
        }
        yield simple_tweet

def filter_lang(tweets: dict, lang: str) -> Generator[dict, None, None]:
    for tweet in tweets:
        if tweet.get('lang', '') == lang:
            yield tweet

def remove_duplicates(tweets: dict) -> Generator[dict, None, None]:
    seen_tweet_ids = set()

    for tweet in tweets:
        if 'id' not in tweet.keys():
            raise RuntimeError("Missing ID.")

        if tweet["id"] in seen_tweet_ids:
            continue

        seen_tweet_ids.add(tweet["id"])
        yield tweet

def text_format(tweets: dict) -> Generator[dict, None, None]:
    for tweet in tweets:
        tweet['text'] = unicodedata.normalize('NFKC', tweet['text'])
        text = tweet["text"]

        treated_text = text.lower() # lowercase all
        treated_text = re.sub(r'\B@\w+', SPECIAL_TOKENS['mention'], treated_text)
        treated_text = re.sub(r'(?:(?:https?):\/\/)?[\w/\-?=%.]+\.[\w/\-?=%.]+',
                              SPECIAL_TOKENS['link'], treated_text)

        tweet["treated_text"] = treated_text
        yield tweet

def dataset_preprocess(filepath: str, lang: str = 'pt') -> Generator[dict, None, None]:
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
    tweets = get_original_posts(tweets)
    if lang:
        tweets = filter_lang(tweets, lang)
    tweets = remove_duplicates(tweets)
    tweets = text_format(tweets)
    yield from tweets


def _serial_dataset_preprocess(all_filepaths, output_filepath, lang):

    with lzma.LZMAFile(output_filepath, mode="wb", format=lzma.FORMAT_XZ) as fout:
        for filename in all_filepaths:
            for tweet in tqdm(dataset_preprocess(filename, lang)):
                fout.write(json.dumps(tweet, separators=(',', ':')).encode('utf-8'))
                fout.write("\n".encode("utf-8"))


def _parallel_dataset_preprocess(all_filepaths, output_filepath, lang):
    # FIXME: check if this breaks deduplication because it has state

    def pickable_dataset_preprocess(*args, **kwargs):
        return list(dataset_preprocess(*args, **kwargs))

    with joblib.Parallel(verbose=60, n_jobs=-1) as parallel:

        tweet_batch = parallel(joblib.delayed(pickable_dataset_preprocess)(filename, lang) for filename in all_filepaths)
        with lzma.LZMAFile(output_filepath, mode="w", format=lzma.FORMAT_XZ) as fout:
            for tweets in tweet_batch:
                for tweet in tqdm(tweets):
                    fout.write(json.dumps(tweet, separators=(',', ':')).encode('utf-8'))
                    fout.write("\n".encode("utf-8"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs='+', help="Path to compacted tweets file(s)", type=str)
    parser.add_argument("-o", "--output", help="Path to output file", type=str, required=True)
    parser.add_argument("-l", "--lang", help="Language", type=str, default="pt")
    args = parser.parse_args()
    _serial_dataset_preprocess(args.input, args.output, args.lang)
