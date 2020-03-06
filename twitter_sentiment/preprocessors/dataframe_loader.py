import pandas as pd
from pandas.api.types import CategoricalDtype
import itertools
from twitter_sentiment.preprocessors import utils, TAGS, CATEGORIES


class DataframeLoader(object):
    """
    Loads lzma file to pandas.
    Tweets are expected to have at least text and tags fields.
    Drops tweets with ambiguous tags and removes tokens from text.

    usage:
    >>> df = DataframeLoader()(filepath)

    :param filepath: file of lmza compressed tweets jsonlines
    :param tags: dictionary of tags and emoticons
    :returns: Dataframe containing text without tokens and polarity column.
    """

    def __call__(self, filepath, tags=TAGS):
        tweets = utils.read_jsonlines_lzma(filepath)
        df = pd.DataFrame.from_dict(tweets)

        # Changes tags for polarities and remove tokens
        df = self.set_polarity(df, tags)
        del df["tags"]
        df = df[df.polarity != "UNDEFINED"]
        df = self.remove_tokens(df, tags)
        return df

    def set_polarity(self, df, tags):
        def get_polarity(tags):
            tags = tuple(tags)

            positive_tags = [('EMOJI_POSITIVE',), ('POSITIVE',), ('POSITIVE', 'EMOJI_POSITIVE')]
            negative_tags = [('EMOJI_NEGATIVE',), ('NEGATIVE',), ('NEGATIVE', 'EMOJI_NEGATIVE')]

            if tags in negative_tags:
                return 'NEGATIVE'
            elif tags in positive_tags:
                return 'POSITIVE'
            else:
                return 'UNDEFINED'

        df["polarity"] = df["tags"].apply(get_polarity)
        polarity_type = CategoricalDtype(categories=CATEGORIES, ordered=True)
        df["polarity"] = df["polarity"].astype(polarity_type)
        return df

    def remove_tokens(self, df, tags):
        tags_tokens = list(itertools.chain(*tags.values()))

        def remove_token_from_string(tweet):
            for token in tags_tokens:
                tweet = tweet.replace(token, '')
            return tweet

        df['text'] = df['text'].apply(remove_token_from_string)
        return df
