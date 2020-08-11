import numpy as np
import tensorflow as tf
from twitter_sentiment.preprocessors import SPECIAL_TOKENS
from transformers import AutoTokenizer, TFBertModel
from sklearn.utils.class_weight import compute_class_weight
from gensim.models import Word2Vec
from tensorflow.keras import models, layers, losses, regularizers, callbacks
from twitter_sentiment.preprocessors.utils import read_jsonlines_lzma
from twitter_sentiment.preprocessors.distant_supervision import extract_tweets_tokenized_text_and_Y



def train_model(filepath: str, model_output: str):

    # loads data
    tweets = read_jsonlines_lzma(filepath)
    tweets, tokenized_texts, Y = extract_tweets_tokenized_text_and_Y(tweets)

    # FIXME: debug
    tokenized_texts = tokenized_texts[0:2048]
    Y = Y[0:2048]
    max_len = 86

    # load pretrained
    tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    bert = TFBertModel.from_pretrained("neuralmind/bert-base-portuguese-cased", from_pt=True)

    # adds our custom tokens
    for k, v in SPECIAL_TOKENS.items():
        tokenizer.add_tokens(v, special_tokens=True)
    bert.resize_token_embeddings(len(tokenizer))

    # freezes bert weights leaving only classifiers active
    bert.trainable = False
    for w in bert.bert.weights:
        w._trainable= False

    # TODO: verificar como as hashtags estao chegando, funcoes uteis: tokenizer.decode(), tokenizer.ids_to_tokens

    X = tokenizer(tokenized_texts, is_pretokenized=True, padding=True, return_tensors='tf')
    X = [X["input_ids"], X["token_type_ids"], X["attention_mask"],]


    input_ids = layers.Input(shape=(None,), dtype=tf.int32)
    token_type_ids = layers.Input(shape=(None,), dtype=tf.int32)
    attention_mask = layers.Input(shape=(None,), dtype=tf.int32)

    embedding = bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[1]

    # FIXME: adds regularizer?
    dense1 = layers.Dense(100, activation="sigmoid")(embedding)
    output = layers.Dense(1, activation="sigmoid")(dense1)

    model = models.Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=[output])


    model.compile(optimizer='adam',
                  loss=losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()

    classes = np.unique(Y).tolist()
    class_weights = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=Y)))

    epochs = 400
    batch_size = 128

    log_dir = "/tmp/tensorboard/"

    model.fit(x=X, y=Y,
              batch_size=batch_size, epochs=epochs, verbose=1,
              validation_split=0.2, shuffle=True, class_weight=class_weights,
              callbacks=[callbacks.EarlyStopping(monitor="loss", min_delta=.0005, patience=3),
                         callbacks.TensorBoard(log_dir=log_dir, write_graph=False)])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to dataset", type=str)
    parser.add_argument("-mo", "--model_output", help="Path to model output file", type=str, required=True)
    args = parser.parse_args()
    train_model(args.input, args.model_output)
