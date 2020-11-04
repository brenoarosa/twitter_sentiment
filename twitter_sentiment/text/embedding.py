import multiprocessing
from typing import Iterable, Tuple, Callable
import numpy as np
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.doc2vec import TaggedDocument
from twitter_sentiment.utils import get_logger
from twitter_sentiment.preprocessors.utils import read_jsonlines_lzma
from twitter_sentiment.preprocessors.tokenizer import tokenize

logger = get_logger()

def train_w2v(filepath: str, output_path: str, tokenizer: Callable = tokenize, **kwargs):

    model_parameters = {
        "epochs": 5,
        "size": 100,
        "window": 5,
        "min_count": 1,
        "max_vocab_size": None
    }

    model_parameters.update(kwargs.get('model_parameters', {}))

    tweets = read_jsonlines_lzma(filepath)
    tweets = tokenizer(tweets)
    logger.info("tokenizing tweets...")
    tweets_tokens = [t["tokenized_treated_text"] for t in tweets] # generator to list

    workers = kwargs.pop('workers', multiprocessing.cpu_count())

    # FIXME compute loss seams to not work
    model = Word2Vec(size=model_parameters["size"], window=model_parameters["window"],
                     min_count=model_parameters["min_count"], max_vocab_size=model_parameters["max_vocab_size"],
                     compute_loss=True, workers=workers)

    logger.info("building vocab...")
    model.build_vocab(tweets_tokens)
    logger.info("training model...")
    model.train(tweets_tokens, total_examples=model.corpus_count, epochs=model_parameters["epochs"])
    model.save(output_path)

def load_w2v_weight_and_X(model: Word2Vec, tweets_tokens: Iterable[list], seq_len: int = 40) -> Tuple[np.ndarray, np.ndarray]:

    word2index = {v: k for k, v in enumerate(model.wv.index2word)}
    X = np.zeros((len(tweets_tokens), seq_len))

    for i, tweet_tokens in enumerate(tweets_tokens):
        for j, token in enumerate(tweet_tokens):
            if j >= seq_len:
                break

            word_idx = word2index.get(token, 0)
            X[i, j] = word_idx

    pad_vector = np.zeros(model.vector_size)
    weights = np.vstack((pad_vector, model.wv.vectors))
    return weights, X

def train_d2v(filepath: str, output_path: str, tokenizer: Callable = tokenize, **kwargs):
    # FIXME: try with nb/svm

    model_parameters = {
        "epochs": 5,
        "vector_size": 100,
        "window": 5,
        "min_count": 1,
        "max_vocab_size": None
    }

    model_parameters.update(kwargs.get('model_parameters', {}))

    workers = kwargs.pop('workers', multiprocessing.cpu_count())

    tweets = read_jsonlines_lzma(filepath)
    tweets = tokenizer(tweets)
    logger.info("tokenizing tweets...")
    tweets_corpus = [TaggedDocument(t["tokenized_treated_text"], [t["id"]]) for t in tweets] # generator to list

    model = Doc2Vec(vector_size=model_parameters["vector_size"], window=model_parameters["window"],
                    min_count=model_parameters["min_count"], max_vocab_size=model_parameters["max_vocab_size"],
                    workers=workers)

    logger.info("building vocab...")
    model.build_vocab(tweets_corpus)
    logger.info("training model...")
    model.train(tweets_corpus, total_examples=model.corpus_count, epochs=model_parameters["epochs"])
    model.save(output_path)

def load_d2v_X(model: Doc2Vec, tweets_tokens: Iterable[list]) -> np.ndarray:

    X = np.zeros((len(tweets_tokens), model.vector_size))
    for i, tweet_tokens in enumerate(tweets_tokens):
        x = model.infer_vector(tweet_tokens)
        X[i, :] = np.array(x)

    return X


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to dataset", type=str)
    parser.add_argument("-o", "--output", help="Path to text embedding output file", type=str, required=True)
    args = parser.parse_args()
    train_w2v(args.input, args.output)
