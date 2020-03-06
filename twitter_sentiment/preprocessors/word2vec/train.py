import json
import os
import glob
from gensim.models import Word2Vec
import multiprocessing

class W2VTrainer(object):

    def __call__(self, tokenized_tweets_path, **kwargs):

        out_path = kwargs.pop('out_path', "/dev/null")
        model_parameters = kwargs.pop('model_parameters', {
                                          "epochs": 1,
                                          "size": 100,
                                          "window": 5,
                                          "min_count": 1,
                                          "max_vocab_size": None
                                      })
        workers = kwargs.pop('workers', 2 * multiprocessing.cpu_count() + 1)

        model = Word2Vec(size=model_parameters["size"], window=model_parameters["window"],
                         min_count=model_parameters["min_count"], max_vocab_size=model_parameters["max_vocab_size"],
                         null_word=0, workers=workers)

        tweets = self.tweet_gen(tokenized_tweets_path)
        model.build_vocab(tweets)

        epochs = model_parameters.pop("epochs", 1)

        for i in range(epochs):
            print("Training epoch {}/{}".format((i+1), epochs))
            tweets = self.tweet_gen(tokenized_tweets_path)
            model.train(tweets, total_examples=model.corpus_count, epochs=1)
            # Saves trainable model
            model.save(out_path)

        # Removes trainable model
        for f in glob.glob(out_path + "*"):
            os.remove(f)

        # Saves final trained model (non-trainable)
        model = model.wv
        model.save(out_path)

        return model

    def tweet_gen(self, tokenized_tweets_path):
        # TODO: ler em memoria pra aumentar o paralelismo
        with open(tokenized_tweets_path) as fin:
            for line in fin:
                data = json.loads(line)
                yield data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to tokenized tweets", type=str)
    parser.add_argument("-mp", "--model_parameters", help="json encoded parameters to be passed to word2vec", type=json.loads)
    parser.add_argument("-o", "--output", help="Path to output model", type=str, required=True)
    args = parser.parse_args()

    W2VTrainer()(args.input, out_path=args.output, model_parameters=args.model_parameters)
