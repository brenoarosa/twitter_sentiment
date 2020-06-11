from typing import Iterable
import csv
import lzma
from tqdm import tqdm
from twitter_sentiment.preprocessors.utils import read_jsonlines_lzma


def get_retweets(tweets: Iterable[dict]) -> Iterable[dict]:
    for tweet in tweets:
        if "retweeted_status" not in tweet.keys():
            continue

        simple_tweet = {
            "user_id": tweet["user_info"]["id_str"],
            "retweeted_user_id": tweet["retweeted_status"]["user"]["id_str"],
            "tweet_id": tweet["id"],
            "retweeted_tweet_id": tweet["retweeted_status"]["id_str"],
        }
        yield simple_tweet

def extract_retweet_users(tweets: Iterable[dict]) -> Iterable[dict]:
    for tweet in tweets:
        tweet = {k: v for k, v in tweet.items() if k in ['user_id', 'retweeted_user_id']}
        yield tweet

def graph_preprocess(filepath: str) -> Iterable[dict]:
    """
    Get retweet edgelist from tweets given filepath.

    usage:
    >>> filepath = '/var/data/tweets.xz'
    >>> edgelist = graph_preprocess(filepath)

    :param filepath: path to lzma compressed tweets
    :returns: generator of edgelist
    """

    # FIXME: check if need to deduplicate (or count) edge pairs
    tweets = read_jsonlines_lzma(filepath)
    tweets = get_retweets(tweets)
    tweets = extract_retweet_users(tweets)
    yield from tweets


def _serial_graph_preprocess(all_filepaths: Iterable[str], output_filepath: str):

    #with lzma.LZMAFile(output_filepath, mode="wb", format=lzma.FORMAT_XZ) as fout:
    with open(output_filepath, mode="w") as fout:
        writer = csv.DictWriter(fout, fieldnames=["user_id", "retweeted_user_id"], quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for filename in all_filepaths:
            for tweet in tqdm(graph_preprocess(filename)):
                writer.writerow(tweet)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs='+', help="Path to compacted tweets file(s)", type=str)
    parser.add_argument("-o", "--output", help="Path to output file", type=str, required=True)
    args = parser.parse_args()
    _serial_graph_preprocess(args.input, args.output)
