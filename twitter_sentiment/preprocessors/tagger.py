from collections import Counter
from typing import Generator
import lzma
import joblib
import json
import glob
from twitter_sentiment.preprocessors import utils, TAGS

class Tagger(object):
    """
    Tag tweets by emoticon given filepath.
    Uses original tweets (no retweets or quotes), filtered by language if available.
    Removes duplications and ignores tweets without tags.

    usage:
    >>> filepath = '/var/data/tweets.xz'
    >>> lang = 'pt'
    >>> tagged_tweets = Tagger()(filepath, lang)

    :param filepath: lzma compressed tweets
    :param lang: language to filter, tweets without language field will be ignored. Default: 'pt'.
    :param tags: dictionary of tags and emoticons
    :returns: generator of tagged tweets
    """

    def __init__(self):
        super(Tagger, self).__init__()
        self.seen_tweets = set()
        self.duplicates = Counter()

    def __call__(self, filepath, lang='pt', tags=TAGS):
        tweets = utils.read_jsonlines_lzma(filepath)
        tweets = self.get_original_posts(tweets)
        if lang:
            tweets = self.filter_lang(tweets, lang)
        tweets = self.remove_duplicates(tweets)
        tweets = self.simplify_tweet(tweets)
        tweets = self.filter_and_tag(tweets, tags)
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

    def simplify_tweet(self, tweets):
        for tweet in tweets:
            yield {
                "id": tweet["id"],
                "created_at": tweet["created_at"],
                "text": tweet["text"],
            }

    def filter_and_tag(self, tweets, tags):
        for tweet in tweets:
            tweet_tags = set()
            for tag, tokens in tags.items():
                if any(x in tweet["text"] for x in tokens):
                    tweet_tags.add(tag)

            if tweet_tags:
                tweet["tags"] = list(tweet_tags)
                yield tweet


def _tag(filename, lang):
    return list(Tagger()(filename, lang))

def _parallel_tag(all_filepaths, output_filepath, lang):

    with joblib.Parallel(verbose=60, n_jobs=-1) as parallel:

        results = parallel(joblib.delayed(_tag)(filename, lang) for filename in all_filepaths)
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

    inputs = []
    # expands glob if necessary
    for filename in args.input:
        inputs += glob.glob(filename)
    _parallel_tag(inputs, args.output, args.lang)
