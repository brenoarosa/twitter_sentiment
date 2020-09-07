import joblib
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer
from twitter_sentiment.utils import identity
from twitter_sentiment.preprocessors.utils import read_jsonlines_lzma
from twitter_sentiment.preprocessors.distant_supervision import extract_tweets_tokenized_text_and_Y

def train_model(filepath: str, model_output: str, vectorizer_output: str):
    tweets = read_jsonlines_lzma(filepath)

    _, X, Y = extract_tweets_tokenized_text_and_Y(tweets)

    vectorizer = CountVectorizer(tokenizer=identity, preprocessor=identity)
    X = vectorizer.fit_transform(X)

    estimator = linear_model.SGDClassifier(
        loss='hinge',
        penalty='l2',
        max_iter=1000,
        tol=1e-2,
        class_weight="balanced")

    model = GridSearchCV(
        estimator=estimator,
        param_grid={
            "alpha": np.logspace(-7, -3, num=50),
        },
        cv=StratifiedKFold(n_splits=10, shuffle=True),
        n_jobs=-1,
        scoring=["roc_auc", "accuracy"],
        refit="roc_auc"
    )

    model.fit(X, Y)

    model = CalibratedClassifierCV(model.best_estimator_, cv=5, method='sigmoid')
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
