import unicodedata
from typing import Union
from pandas import DataFrame

def tweet_format(tweet: Union[str, DataFrame]) -> Union[str, DataFrame]:
    """
    Normalizes unicode.
    Trims messages longer than 140 characters.

    usage:
    >>> tweets_df = tweet_format(tweets_df)

    :param tweet: tweet text or dataframe containing tweets (mandatory 'text' column)
    :returns: formatted text or dataframe
    """
    if isinstance(tweet, pd.DataFrame):
        tweet['text'] = tweet["text"].apply(lambda text: unicodedata.normalize('NFKC', text))
        tweet["text"] = tweet["text"].apply(lambda text: text[0:140])
        return tweet

    tweet = unicodedata.normalize('NFKC', tweet)
    tweet = tweet[0:140]
    return tweet
