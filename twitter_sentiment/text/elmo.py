import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from gensim.models import Word2Vec
from tensorflow.keras import models, layers, losses, regularizers, callbacks
from twitter_sentiment.preprocessors.utils import read_jsonlines_lzma
from twitter_sentiment.text.embedding import load_w2v_weight_and_X
from twitter_sentiment.preprocessors.distant_supervision import extract_tweets_tokenized_text_and_Y
from allennlp.commands.elmo import ElmoEmbedder

MAX_LEN = 50
lens = []

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
        batch_Y = [self.Y[i] for i in indexes]

        # TODO: precisa forcar o tamanho pra um valor maximo pro shape ficar igual https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
        # cuidado que esse pad e 2d, outra opcao iniciar outro tensor zerado de tamanho fixo
        pt_tensor, _ = self.elmo_embedder.batch_to_embeddings(batch_tokenized_texts)
        batch_size, elmo_outputs, seq_len, emb_dim = pt_tensor.shape
        # concatenate all emb_dimensions / maybe use only top layer (emb[-1]) https://github.com/allenai/allennlp/blob/v0.9.0/allennlp/commands/elmo.py#L380
        pt_tensor = pt_tensor.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        batch_X = pt_tensor.numpy()

        lens.append(seq_len)
        fixed_size_batch_X = np.zeros((batch_size, MAX_LEN, elmo_outputs*emb_dim))
        fixed_size_batch_X[0:batch_size, 0:MAX_LEN, 0:elmo_outputs*emb_dim] = batch_X[0:batch_size, 0:MAX_LEN, 0:elmo_outputs*emb_dim]

        return fixed_size_batch_X, batch_Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.Y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

def train_model(filepath: str):
    tweets = read_jsonlines_lzma(filepath)

    tweets, tokenized_texts, Y = extract_tweets_tokenized_text_and_Y(tweets)
    tokenized_texts = pad_elmo_tokens(tokenized_texts)

    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pt/wikipedia/options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pt/wikipedia/elmo_pt_weights.hdf5"
    embedder = ElmoEmbedder(options_file=options_file, weight_file=weight_file, cuda_device=-1) # run on cpu

    tokenized_texts_train, tokenized_texts_test, y_train, y_test = train_test_split(tokenized_texts, Y, test_size=0.20, random_state=42)

    train_data_gen = TweetDataGenerator(tokenized_texts_train, y_train, embedder, batch_size=32, shuffle=True)
    test_data_gen = TweetDataGenerator(tokenized_texts_test, y_test, embedder, batch_size=32, shuffle=True)

    from datetime import datetime

    import ipdb; ipdb.set_trace()

    for i in range(100):
        train_data_gen[i]
        print(datetime.now())

    return


    pt_tensor, _ = embedder.batch_to_embeddings(tokenized_texts[0:512]) # FIXME: huge memory constrain, maybe a dataloader?
    batch_size, elmo_outputs, seq_len, emb_dim = pt_tensor.shape
    # concatenate all emb_dimensions
    pt_tensor = pt_tensor.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
    X = pt_tensor.numpy()
    return


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
    #parser.add_argument("-e", "--embedding", help="Path to saved embedding file", type=str, required=True)
    #parser.add_argument("-mo", "--model_output", help="Path to model output file", type=str, required=True)
    args = parser.parse_args()
    train_model(args.input)