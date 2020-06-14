from typing import Iterable, Tuple
import joblib
import numpy as np
import sklearn.naive_bayes
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from twitter_sentiment.utils import identity
from twitter_sentiment.preprocessors import POSITIVE_CLASS, NEGATIVE_CLASS
from twitter_sentiment.preprocessors.utils import read_jsonlines_lzma

ACCEPTED_CLASSES = [POSITIVE_CLASS, NEGATIVE_CLASS]

def extract_features(tweets: Iterable[dict]) -> Tuple[list, list]:
    X = list()
    Y = list()
    for t in tweets:
        if t["distant_supervision_tags"] not in ACCEPTED_CLASSES:
            continue

        if len(t["tokenized_treated_text"]) == 0:
            continue

        x = t["tokenized_treated_text"]
        y = int(t["distant_supervision_tags"] == POSITIVE_CLASS)
        X.append(x)
        Y.append(y)

    Y = np.array(Y)
    return X, Y

def train_model(filepath: str, model_output: str, vectorizer_output: str):
    tweets = read_jsonlines_lzma(filepath)

    X, Y = extract_features(tweets)

    vectorizer = CountVectorizer(tokenizer=identity, preprocessor=identity)
    X = vectorizer.fit_transform(X)

    model = GridSearchCV(
        estimator=sklearn.naive_bayes.BernoulliNB(fit_prior=False),
        param_grid={
            "alpha": np.logspace(-2, 2, num=50),
        },
        cv=StratifiedKFold(n_splits=10, shuffle=True),
        n_jobs=-1,
        scoring=["roc_auc", "accuracy"],
        refit="roc_auc"
    )

    model.fit(X, Y)
    joblib.dump(vectorizer, vectorizer_output)
    joblib.dump(model, model_output)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to dataset", type=str)
    parser.add_argument("-mo", "--model_output", help="Path to model output file", type=str, required=True)
    parser.add_argument("-vo", "--vectorizer_output", help="Path to vectorizer output file", type=str, required=True)
    args = parser.parse_args()
    train_model(args.input, args.model_output, args.vectorizer_output)
