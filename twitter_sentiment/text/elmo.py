from datetime import datetime
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, losses, regularizers, callbacks
from allennlp.commands.elmo import ElmoEmbedder
from twitter_sentiment.preprocessors.utils import read_jsonlines_lzma
from twitter_sentiment.preprocessors.distant_supervision import extract_tweets_tokenized_text_and_Y
from twitter_sentiment.text import elmo_tokens_len

LOG_DIR = f"logs/tensorboard/elmo/{datetime.now().isoformat()}"

def pad_elmo_tokens(tokenized_texts: list) -> list:
    """
    check https://github.com/allenai/allennlp/blob/v0.9.0/tutorials/how_to/elmo.md
    """
    for i in range(len(tokenized_texts)):
        tokenized_texts[i].insert(0, "<S>")
        tokenized_texts[i].append("</S>")

    return tokenized_texts


class TweetDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, tokenized_texts, Y, elmo_embedder, batch_size=32, shuffle=True):
        self.tokenized_texts = tokenized_texts
        self.Y = Y

        self.elmo_embedder = elmo_embedder

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        return int(len(self.Y) // self.batch_size)

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        batch_tokenized_texts = [self.tokenized_texts[i] for i in indexes]
        batch_Y = np.array([self.Y[i] for i in indexes])

        pt_tensor, _ = self.elmo_embedder.batch_to_embeddings(batch_tokenized_texts)
        batch_size, elmo_outputs, seq_len, emb_dim = pt_tensor.shape

        # if choose to use all layers: concatenate all emb_dimensions
        # pt_tensor = pt_tensor.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        pt_tensor = pt_tensor[:, -1, :, :]

        batch_X = pt_tensor.numpy()
        seq_len = min(seq_len, elmo_tokens_len)

        fixed_size_batch_X = np.zeros((batch_size, elmo_tokens_len, emb_dim))
        fixed_size_batch_X[0:batch_size, 0:seq_len, 0:emb_dim] = batch_X[0:batch_size, 0:seq_len, 0:emb_dim]

        return fixed_size_batch_X, batch_Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.Y))
        if self.shuffle:
            np.random.shuffle(self.indexes)


def train_model(filepath: str, model_output: str):
    tweets = read_jsonlines_lzma(filepath)

    tweets, tokenized_texts, Y = extract_tweets_tokenized_text_and_Y(tweets)
    #tweets = tweets[0:10000]; tokenized_texts = tokenized_texts[0:10000]; Y = Y[0:10000]
    tokenized_texts = pad_elmo_tokens(tokenized_texts)

    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pt/wikipedia/options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pt/wikipedia/elmo_pt_weights.hdf5"
    embedder = ElmoEmbedder(options_file=options_file, weight_file=weight_file, cuda_device=-1) # run on cpu

    tokenized_texts_train, tokenized_texts_test, y_train, y_test = train_test_split(tokenized_texts, Y, test_size=0.20, random_state=42)

    train_data_gen = TweetDataGenerator(tokenized_texts_train, y_train, embedder, batch_size=64, shuffle=True)
    test_data_gen = TweetDataGenerator(tokenized_texts_test, y_test, embedder, batch_size=64, shuffle=True)

    text_input = tf.keras.Input(shape=(elmo_tokens_len, 1024), name="elmo_embeddings")

    text_conv_layer_3 = layers.Conv1D(filters=100, kernel_size=3, padding="same",
                                      kernel_regularizer=regularizers.l2(10**-4),
                                      bias_regularizer=regularizers.l2(10**-4),
                                      name="text_conv_3")(text_input)
    text_conv_layer_4 = layers.Conv1D(filters=100, kernel_size=4, padding="same",
                                      kernel_regularizer=regularizers.l2(10**-4),
                                      bias_regularizer=regularizers.l2(10**-4),
                                      name="text_conv_4")(text_input)
    text_conv_layer_5 = layers.Conv1D(filters=100, kernel_size=5, padding="same",
                                      kernel_regularizer=regularizers.l2(10**-4),
                                      bias_regularizer=regularizers.l2(10**-4),
                                      name="text_conv_5")(text_input)

    text_conv_layer = layers.Concatenate()([text_conv_layer_3, text_conv_layer_4, text_conv_layer_5])

    text_conv_layer = layers.BatchNormalization()(text_conv_layer)
    text_conv_layer = layers.Activation("relu")(text_conv_layer)
    text_conv_layer = layers.MaxPooling1D(3)(text_conv_layer)
    text_conv_layer = layers.Dropout(.5)(text_conv_layer)

    text_flat_emb_layer = layers.Flatten(name="text_emb_flat")(text_conv_layer)

    predictions = layers.Dense(1, activation="sigmoid",
                               kernel_regularizer=regularizers.l2(10**-4),
                               bias_regularizer=regularizers.l2(10**-4),
                               name="prediction")(text_flat_emb_layer)

    model = tf.keras.Model(text_input, predictions)
    model.compile(optimizer='adam',
                  loss=losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()

    classes = np.unique(Y).tolist()
    class_weights = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=Y)))

    epochs = 5
    model.fit(x=train_data_gen,
              validation_data=test_data_gen,
              class_weight=class_weights,
              epochs=epochs,
              verbose=1,
              workers=4,
              max_queue_size=20,
              use_multiprocessing=True,
              callbacks=[callbacks.EarlyStopping(monitor="loss", min_delta=.0005, patience=3),
                         callbacks.ModelCheckpoint(model_output, monitor='val_loss', verbose=1,
                                                   save_best_only=True, save_weights_only=False),
                         callbacks.TensorBoard(log_dir=LOG_DIR, write_graph=False)])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to dataset", type=str)
    parser.add_argument("-mo", "--model_output", help="Path to model output file", type=str, required=True)
    args = parser.parse_args()
    train_model(args.input, args.model_output)
