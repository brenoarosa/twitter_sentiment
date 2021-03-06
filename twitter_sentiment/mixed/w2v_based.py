from datetime import datetime
import numpy as np
import joblib
from sklearn.utils.class_weight import compute_class_weight
from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow.keras import models, layers, losses, regularizers, callbacks
from twitter_sentiment.utils import get_logger
from twitter_sentiment.preprocessors.utils import read_jsonlines_lzma
from twitter_sentiment.text import max_tokens_len
from twitter_sentiment.text.embedding import load_w2v_weight_and_X
from twitter_sentiment.graph.embedding import GraphEmbedding, load_graph_emb_weight_and_X
from twitter_sentiment.preprocessors.distant_supervision import extract_tweets_tokenized_text_and_Y

logger = get_logger()

LOG_DIR = f"logs/tensorboard/mixed/{datetime.now().isoformat()}"

def train_model(filepath: str, text_embedding_path: str, text_model_path: str,
                user_embedding_path: str, model_output: str):
    tweets = read_jsonlines_lzma(filepath)

    tweets, tokenized_texts, Y = extract_tweets_tokenized_text_and_Y(tweets)

    # FIXME: fixed to word2vec
    seq_len = max_tokens_len
    w2v = Word2Vec.load(text_embedding_path)
    _, text_X = load_w2v_weight_and_X(w2v, tokenized_texts, seq_len=seq_len)


    words_in_emb = (text_X != 0).mean()
    logger.info(f"percentage of words in embedding: {words_in_emb:.1%}")

    graph_emb = joblib.load(user_embedding_path)
    user_emb_weights, user_X = load_graph_emb_weight_and_X(graph_emb, tweets)

    users_in_emb = (user_X != 0).mean()
    logger.info(f"percentage of users in embedding: {users_in_emb:.1%}")

    # user layers
    user_input = tf.keras.Input(shape=(1,), name="user_input")

    user_emb_layer = layers.Embedding(user_emb_weights.shape[0], user_emb_weights.shape[1],
                                      input_length=1, weights=[user_emb_weights],
                                      trainable=False, name="user_embbedding")(user_input)

    user_flat_emb_layer = layers.Flatten(name="user_emb_flat")(user_emb_layer)

    # text layers

    text_model = models.load_model(text_model_path)
    text_model.trainable = True

    text_input = text_model.input
    text_flat_emb_layer = text_model.layers[-2].output # TODO: CHECK TO VALIDATE IF WORKS IN LSTM

    # joint layers

    composed_emb = layers.Concatenate(name="composed")([user_flat_emb_layer, text_flat_emb_layer])
    dense_layer = layers.Dense(128, activation="relu", name="joint_dense_1")(composed_emb)
    dense_layer = layers.Dense(32, activation="relu", name="joint_dense_2")(dense_layer)
    predictions = layers.Dense(1, activation="sigmoid", name="prediction")(dense_layer)

    model = tf.keras.Model([user_input, text_input], predictions)
    model.compile(optimizer='adam',
                  loss=losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()

    classes = np.unique(Y).tolist()
    class_weights = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=Y)))

    epochs = 400
    batch_size = 2048

    model.fit(x=[user_X, text_X], y=Y,
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
    parser.add_argument("-te", "--text_embedding", help="Path to saved word2vec file", type=str, required=True)
    parser.add_argument("-tm", "--text_model", help="Path to saved text model file", type=str, required=True)
    parser.add_argument("-ue", "--user_embedding", help="Path to saved user embedding file", type=str, required=True)
    parser.add_argument("-mo", "--model_output", help="Path to model output file", type=str, required=True)
    args = parser.parse_args()
    train_model(args.input, args.text_embedding, args.text_model, args.user_embedding, args.model_output)
