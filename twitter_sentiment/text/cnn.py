import tensorflow as tf
from tensorflow.keras import models, layers, losses, regularizers


def train_model(filepath: str, model_output: str):

    embedding_w = tf.random.normal([10000, 200], 0, 1, tf.float32, seed=1)

    model = models.Sequential()
    model.add(layers.Embedding(10000, 200,
                               input_length=40,
                               weights=[embedding_w],
                               trainable=False))

    model.add(layers.Conv1D(filters=100, kernel_size=3, padding="same",
                            kernel_regularizer=regularizers.l2(10**-3),
                            bias_regularizer=regularizers.l2(10**-3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Dropout(.5))

    model.add(layers.Flatten())
    model.add(layers.Dense(1,
                           kernel_regularizer=regularizers.l2(10**-3),
                           bias_regularizer=regularizers.l2(10**-3)))
    model.compile(optimizer='adam',
                  loss=losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to dataset", type=str)
    parser.add_argument("-mo", "--model_output", help="Path to model output file", type=str, required=True)
    args = parser.parse_args()
    train_model(args.input, args.model_output)
