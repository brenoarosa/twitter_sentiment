import nltk
from operator import methodcaller
from nltk.corpus import stopwords

class TwitterTokenizer(object):
    """
    Custom analyzer to be applied to Scikit learn CountVectorizer.
    """

    def __init__(self, reduce_len=True, lang="pt"):
        lang_translater = {"pt": "portuguese", "en": "english"}
        self.tokenizer = nltk.tokenize.casual.TweetTokenizer(reduce_len=reduce_len)
        self.stop = set(stopwords.words(lang_translater[lang]))

    def __call__(self, text, remove_stopwords=True, remove_single=True):
        tokens = self.tokenizer.tokenize(text)

        # Lower case and filter tokens
        tokens = map(lambda x: x.lower(), tokens)
        tokens = filter(lambda x: x[0] != '@', tokens)
        tokens = filter(lambda x: not x.startswith("http://"), tokens)
        tokens = filter(lambda x: not x.startswith("https://"), tokens)

        if remove_single:
            tokens = filter(lambda x: len(x) >= 2, tokens)
        if remove_stopwords:
            tokens = filter(lambda x: x not in self.stop, tokens)
        return tokens
