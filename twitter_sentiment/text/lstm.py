import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from gensim.models import Word2Vec
from tensorflow.keras import models, layers, losses, regularizers, callbacks
from twitter_sentiment.preprocessors.utils import read_jsonlines_lzma
from twitter_sentiment.text.embedding import load_w2v_weight_and_X
from twitter_sentiment.preprocessors.distant_supervision import extract_tweets_tokenized_text_and_Y


def train_model(filepath: str, embedding_path: str, model_output: str):
    tweets = read_jsonlines_lzma(filepath)

    tweets, tokenized_texts, Y = extract_tweets_tokenized_text_and_Y(tweets)

    seq_len = 40 # FIXME
    w2v = Word2Vec.load(embedding_path)
    emb_weights, X = load_w2v_weight_and_X(w2v, tokenized_texts, seq_len=seq_len)

    model = models.Sequential()
    model.add(layers.Embedding(emb_weights.shape[0], emb_weights.shape[1],
                               input_length=seq_len,
                               weights=[emb_weights],
                               trainable=False))

    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
    model.add(layers.Bidirectional(layers.LSTM(64)))

    model.add(layers.Dropout(.5))

    model.add(layers.Dense(1,
                           activation="sigmoid",
                           kernel_regularizer=regularizers.l2(10**-3),
                           bias_regularizer=regularizers.l2(10**-3)))
    model.compile(optimizer='adam',
                  loss=losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()

    classes = np.unique(Y).tolist()
    class_weights = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=Y)))

    epochs = 400
    batch_size = 2048

    log_dir = "/tmp/tensorboard/"

    model.fit(x=X, y=Y,
              batch_size=batch_size, epochs=epochs, verbose=1,
              validation_split=0.2, shuffle=True, class_weight=class_weights,
              callbacks=[callbacks.EarlyStopping(monitor="loss", min_delta=.0005, patience=3),
                         callbacks.ModelCheckpoint(model_output, monitor='val_loss', verbose=1,
                                                   save_best_only=True, save_weights_only=False),
                         callbacks.TensorBoard(log_dir=log_dir, write_graph=False)])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to dataset", type=str)
    parser.add_argument("-e", "--embedding", help="Path to saved embedding file", type=str, required=True)
    parser.add_argument("-mo", "--model_output", help="Path to model output file", type=str, required=True)
    args = parser.parse_args()
    train_model(args.input, args.embedding, args.model_output)
