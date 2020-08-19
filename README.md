# Twitter Sentiment Classifier

Code implementing my [master thesis](https://github.com/brenoarosa/thesis).

## Dataset Pre-Processing
Extract core tweet fields from raw tweets:
```sh
python -m twitter_sentiment.preprocessors.dataset_preprocess data/tweets/amazonia_lzma/*.jsonlines.xz -o data/output/amazonia-pt.jsonline.xz -l pt
```

Tag with distant supervision:
```sh
python -m twitter_sentiment.preprocessors.distant_supervision data/output/amazonia-pt.jsonline.xz -o data/output/amazonia-pt-tagged.jsonline.xz -l pt
```

Build edgelist from raw tweets:
```sh
python -m twitter_sentiment.graph.preprocess data/tweets/amazonia_lzma/*.jsonlines.xz -o data/output/amazonia-edgelist.csv
```

Calculate base graph stats:
```sh
python -m twitter_sentiment.graph.stats data/output/amazonia-edgelist.csv -o data/output/amazonia-graph-stats.json
```

## Text Classification
Train Naive Bayes classifier:
```sh
python -m twitter_sentiment.text.nb data/output/amazonia-pt-tagged.jsonline.xz -mo models/amazonia-pt-nb.pickle -vo models/amazonia-pt-nb-vectorizer.pickle
```

Train SVM classifier:
```sh
python -m twitter_sentiment.text.svm data/output/amazonia-pt-tagged.jsonline.xz -mo models/amazonia-pt-svm.pickle -vo models/amazonia-pt-svm-vectorizer.pickle
```

Train W2V embedding:
```sh
python -m twitter_sentiment.text.embedding data/output/amazonia-pt.jsonline.xz -o models/amazonia-pt-w2v.emb
```

Train CNN + W2V classifier:
```sh
python -m twitter_sentiment.text.cnn data/output/amazonia-pt-tagged.jsonline.xz -e models/amazonia-pt-w2v.emb -mo models/amazonia-pt-cnn.h5
```

## Graph Classification
Get graph embeddings:
```sh
# node2vec
python -m twitter_sentiment.graph.embedding data/output/all-edgelist.csv -a node2vec -o models/all-graph-embedding-lle.emb

# lle
python -m twitter_sentiment.graph.embedding data/output/all-edgelist.csv -a lle -o models/all-graph-embedding-lle.emb
```

## Joint Classification
CNN + Node2Vec:
```sh
python -m twitter_sentiment.mixed.cnn_node2vec data/output/amazonia-pt-tagged.jsonline.xz -te models/amazonia-pt-w2v.emb -ue models/amazonia-graph-embedding.emb -mo models/amazonia-pt-cnn_node2vec.h5
```
