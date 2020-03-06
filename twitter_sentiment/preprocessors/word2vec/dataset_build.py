import lzma
import json
import nltk
from tqdm import tqdm
from nltk.corpus import stopwords
from twitter_sentiment.analyzer import TwitterAnalyzer
from twitter_sentiment.preprocessors import tweet_format
from twitter_sentiment.preprocessors.utils import read_jsonlines_lzma

class DatasetBuild(object):

    def __init__(self, reduce_len=True, lang="pt"):
        self.analyzer = TwitterAnalyzer(reduce_len, lang)

    def __call__(self, in_path, out_path):

        num_lines = sum(1 for line in open(in_path))
        with open(out_path, mode='w') as fout:
            with open(in_path, mode='r') as fin:
                for line in tqdm(fin, total=num_lines):
                    text = json.loads(line)
                    text = tweet_format(text)
                    tokens = self.analyzer(text, remove_stopwords=True, remove_single=False)
                    data = json.dumps(list(tokens))
                    fout.write(data)
                    fout.write("\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to decompressed tweets text file(s)", type=str)
    parser.add_argument("-o", "--output", help="Path to output file", type=str, required=True)
    parser.add_argument("-l", "--lang", help="Language", type=str, default="pt")
    args = parser.parse_args()
    DatasetBuild(lang=args.lang)(args.input, args.output)
