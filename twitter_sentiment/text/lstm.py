from datetime import datetime
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow.keras import models, layers, losses, regularizers, callbacks
from twitter_sentiment.preprocessors.utils import read_jsonlines_lzma
from twitter_sentiment.text import max_tokens_len
from twitter_sentiment.text.embedding import load_w2v_weight_and_X
from twitter_sentiment.preprocessors.distant_supervision import extract_tweets_tokenized_text_and_Y

LOG_DIR = f"logs/tensorboard/lstm/{datetime.now().isoformat()}"

def train_model(filepath: str, embedding_path: str, model_output: str):
    tweets = read_jsonlines_lzma(filepath)

    tweets, tokenized_texts, Y = extract_tweets_tokenized_text_and_Y(tweets)

    seq_len = max_tokens_len
    w2v = Word2Vec.load(embedding_path)
    text_emb_weights, X = load_w2v_weight_and_X(w2v, tokenized_texts, seq_len=seq_len)

    text_input = tf.keras.Input(shape=(seq_len,), name="text_input")

    text_emb_layer = layers.Embedding(text_emb_weights.shape[0], text_emb_weights.shape[1],
                                      input_length=seq_len, weights=[text_emb_weights],
                                      trainable=False, name="text_embbedding")(text_input)

    lstm_layer_1 = layers.Bidirectional(layers.LSTM(128, return_sequences=True), name="lstm_layer_1")(text_emb_layer)
    lstm_layer_2 = layers.Bidirectional(layers.LSTM(128), name="lstm_layer_2")(lstm_layer_1)

    dropout = layers.Dropout(.5)(lstm_layer_2)
    predictions = layers.Dense(1, activation="sigmoid",
                               kernel_regularizer=regularizers.l2(10**-4),
                               bias_regularizer=regularizers.l2(10**-4),
                               name="prediction")(dropout)

    model = tf.keras.Model(text_input, predictions)
    model.compile(optimizer='adam',
                  loss=losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()

    classes = np.unique(Y).tolist()
    class_weights = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=Y)))

    epochs = 400
    batch_size = 2048

    model.fit(x=X, y=Y,
              batch_size=batch_size, epochs=epochs, verbose=1,
              validation_split=0.2, shuffle=True, class_weight=class_weights,
              callbacks=[callbacks.EarlyStopping(monitor="loss", min_delta=.0005, patience=3),
                         callbacks.ModelCheckpoint(model_output, monitor='val_loss', verbose=1,
                                                   save_best_only=True, save_weights_only=False),
                         callbacks.TensorBoard(log_dir=LOG_DIR, write_graph=False)])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to dataset", type=str)
    parser.add_argument("-e", "--embedding", help="Path to saved embedding file", type=str, required=True)
    parser.add_argument("-mo", "--model_output", help="Path to model output file", type=str, required=True)
    args = parser.parse_args()
    train_model(args.input, args.embedding, args.model_output)
