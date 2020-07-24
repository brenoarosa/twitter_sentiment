from typing import Iterable
import json
import lzma
import re
import unicodedata
import joblib
from tqdm import tqdm
from twitter_sentiment.preprocessors import SPECIAL_TOKENS
from twitter_sentiment.preprocessors.utils import read_jsonlines_lzma

def get_original_posts_v1(tweets: Iterable[dict]) -> Iterable[dict]:
    for tweet in tweets:
        user_field = "user"
        text_field = "full_text"
        id_field = "id_str"

        if "error" in tweet.keys():
            continue

        # use quoted tweets same as original
        if "retweeted_status" in tweet.keys():
            tweet = tweet["retweeted_status"]

        simple_tweet = {
            "user_id": tweet[user_field]["id_str"],
            "text": tweet[text_field],
            "id": tweet[id_field],
            "lang": tweet["lang"],
        }
        yield simple_tweet

def get_original_posts_v2(tweets: Iterable[dict]) -> Iterable[dict]:
    for tweet in tweets:
        user_field = "user_info"
        text_field = "text"
        id_field = "id"

        if "error" in tweet.keys():
            continue

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

def get_original_posts_best_effort(tweets: Iterable[dict]) -> Iterable[dict]:
    for tweet in tweets:

        if "error" in tweet.keys():
            continue

        # use quoted tweets same as original
        if "retweeted_status" in tweet.keys():
            tweet = tweet["retweeted_status"]

        user = tweet.get("user_info") or tweet["user"]

        simple_tweet = {
            "user_id": user.get("id_str") or user["id"],
            "text": tweet.get("text") or tweet["full_text"],
            "id": tweet.get("id_str") or tweet["id"],
            "lang": tweet["lang"],
        }
        yield simple_tweet

def filter_lang(tweets: Iterable[dict], lang: str) -> Iterable[dict]:
    for tweet in tweets:
        if tweet.get('lang', '') == lang:
            yield tweet

def remove_duplicates(tweets: Iterable[dict]) -> Iterable[dict]:
    seen_tweet_ids = set()

    for tweet in tweets:
        if 'id' not in tweet.keys():
            raise RuntimeError("Missing ID.")

        if tweet["id"] in seen_tweet_ids:
            continue

        seen_tweet_ids.add(tweet["id"])
        yield tweet

def text_format(tweets: Iterable[dict]) -> Iterable[dict]:
    for tweet in tweets:
        tweet['text'] = unicodedata.normalize('NFKC', tweet['text'])
        text = tweet["text"]

        treated_text = text.lower() # lowercase all
        treated_text = re.sub(r'\B@\w+', SPECIAL_TOKENS['mention'], treated_text)
        treated_text = re.sub(r'https:\/\/t.co\/([-a-zA-Z0-9@:%_\+.~#?&//=\]\[]*)', SPECIAL_TOKENS['link'], treated_text)

        tweet["treated_text"] = treated_text
        yield tweet

def dataset_preprocess(filepath: str, lang: str = 'pt') -> Iterable[dict]:
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

    # amazonia uses get_original_posts_v2
    # others mostly uses get_original_posts_v1
    # either failing the get_original_posts_best_effort or any adaptation can be tested

    tweets = read_jsonlines_lzma(filepath)
    tweets = get_original_posts_best_effort(tweets)
    if lang:
        tweets = filter_lang(tweets, lang)
    tweets = remove_duplicates(tweets)
    tweets = text_format(tweets)
    yield from tweets


def _serial_dataset_preprocess(all_filepaths: Iterable[str], output_filepath: str, lang: str):

    with lzma.LZMAFile(output_filepath, mode="wb", format=lzma.FORMAT_XZ) as fout:
        for filename in all_filepaths:
            for tweet in tqdm(dataset_preprocess(filename, lang)):
                fout.write(json.dumps(tweet, separators=(',', ':')).encode('utf-8'))
                fout.write("\n".encode("utf-8"))


def _parallel_dataset_preprocess(all_filepaths: Iterable[str], output_filepath: str, lang: str):
    def pickable_dataset_preprocess(*args, **kwargs):
        return list(dataset_preprocess(*args, **kwargs))

    def deduplicate_tweets(tweets):
        seen_tweet_ids = set()

        for tweet in tweets:
            if tweet["id"] not in seen_tweet_ids:
                seen_tweet_ids.add(tweet["id"])
                yield tweet

    with joblib.Parallel(verbose=60, n_jobs=-1) as parallel:

        tweet_batch = parallel(joblib.delayed(pickable_dataset_preprocess)(filename, lang) for filename in all_filepaths)

        with lzma.LZMAFile(output_filepath, mode="w", format=lzma.FORMAT_XZ) as fout:
            for tweets in tweet_batch:
                tweets = deduplicate_tweets(tweets)
                for tweet in tweets:
                    fout.write(json.dumps(tweet, separators=(',', ':')).encode('utf-8'))
                    fout.write("\n".encode("utf-8"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs='+', help="Path to compacted tweets file(s)", type=str)
    parser.add_argument("-o", "--output", help="Path to output file", type=str, required=True)
    parser.add_argument("-l", "--lang", help="Language", type=str, default="pt")
    args = parser.parse_args()
    _parallel_dataset_preprocess(args.input, args.output, args.lang)
