from collections import Counter
from typing import Generator
import lzma
import joblib
import json
from twitter_sentiment.preprocessors import utils, TAGS

class TextExtractor(object):
    """
    Get text from tweets given filepath.
    Uses original tweets (no retweets or quotes), filtered by language if available.
    Removes duplications.

    usage:
    >>> filepath = '/var/data/tweets.xz'
    >>> lang = 'pt'
    >>> tweets = TextExtractor()(filepath, lang)

    :param filepath: lzma compressed tweets
    :param lang: language to filter, tweets without language field will be ignored. Default: 'pt'.
    :returns: generator of tweets
    """

    def __init__(self):
        super(TextExtractor, self).__init__()
        self.seen_tweets = set()
        self.duplicates = Counter()

    def __call__(self, filepath, lang='pt'):
        tweets = utils.read_jsonlines_lzma(filepath)
        tweets = self.get_original_posts(tweets)
        if lang:
            tweets = self.filter_lang(tweets, lang)
        tweets = self.remove_duplicates(tweets)
        tweets = self.get_text(tweets)
        yield from tweets

    def get_original_posts(self, tweets):
        for tweet in tweets:
            if "quoted_status" in tweet:
                yield tweet["quoted_status"]
            if "retweeted_status" in tweet:
                yield tweet["retweeted_status"]
            else:
                yield tweet

    def filter_lang(self, tweets, lang):
        for tweet in tweets:
            if tweet.get('lang', '') == lang:
                yield tweet

    def remove_duplicates(self, tweets):
        for tweet in tweets:
            if 'id' not in tweet:
                continue

            if tweet["id"] not in self.seen_tweets:
                self.seen_tweets.add(tweet["id"])
                yield tweet

            else:
                self.duplicates[tweet["id"]] += 1

    def get_text(self, tweets):
        for tweet in tweets:
            yield tweet["text"]



def _get_text(filename, lang):
    return list(TextExtractor()(filename, lang))

def _parallel_get_text(all_filepaths, output_filepath, lang):

    with joblib.Parallel(verbose=60, n_jobs=-1) as parallel:

        results = parallel(joblib.delayed(_get_text)(filename, lang) for filename in all_filepaths)
        with lzma.LZMAFile(output_filepath, mode="w", format=lzma.FORMAT_XZ) as fout:
            for tweets in results:
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
    _parallel_get_text(args.input, args.output, args.lang)
