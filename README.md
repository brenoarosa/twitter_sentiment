# Twitter Sentiment Classifier

Code implementing my [master thesis](https://github.com/brenoarosa/thesis).

## Dataset Pre-Processing
Extract core tweet fields from raw tweets:
```sh
python -m twitter_sentiment.preprocessors.dataset_preprocess data/tweets/amazonia_lzma/*.jsonlines.xz -o data/output/amazonia-pt.jsonline.xz -l pt
```

Tag with distant supervision:
```sh
python -m twitter_sentiment.preprocessors.distant_supervision data/output/all_deduplicated-pt.jsonline.xz -o data/output/all_deduplicated-pt-tagged.jsonline.xz -l pt
```

Build edgelist from raw tweets:
```sh
python -m twitter_sentiment.graph.preprocess data/tweets/amazonia_lzma/*.jsonlines.xz -o data/output/amazonia-edgelist.csv
```

Calculate base graph stats:
```sh
python -m twitter_sentiment.graph.stats data/output/all-edgelist.csv -o data/output/all-graph-stats.json
```

## Text Classification
Train Naive Bayes classifier:
```sh
python -m twitter_sentiment.text.nb data/output/all_deduplicated-pt-tagged.jsonline.xz -mo models/nb-pt.pickle -vo models/nb-pt-vectorizer.pickle
```

Train SVM classifier:
```sh
python -m twitter_sentiment.text.svm data/output/all_deduplicated-pt-tagged.jsonline.xz -mo models/svm-pt.pickle -vo models/svm-pt-vectorizer.pickle
```

Train W2V embedding:
```sh
python -m twitter_sentiment.text.embedding data/output/all_deduplicated-pt.jsonline.xz -o models/w2v-pt.emb
```

Train CNN + W2V classifier:
```sh
python -m twitter_sentiment.text.cnn data/output/all_deduplicated-pt-tagged.jsonline.xz -e models/w2v-pt.emb -mo models/cnn-pt.h5
```

Train ELMo:
```sh
python -m twitter_sentiment.text.elmo data/output/all_deduplicated-pt-tagged.jsonline.xz -mo models/elmo-cnn-pt.h5
```

## Graph Representation
Get graph embeddings:
```sh
# node2vec
python -m twitter_sentiment.graph.embedding data/output/all-edgelist.csv -a node2vec -o models/graph-n2v.emb

# lle
python -m twitter_sentiment.graph.embedding data/output/all-edgelist.csv -a lle -o models/graph-lle.emb
```

## Joint Classification
CNN + Node2Vec:
```sh
python -m twitter_sentiment.mixed.cnn_node2vec data/output/all_deduplicated-pt-tagged.jsonline.xz -te models/w2v-pt.emb -ue models/graph-n2v.emb -mo models/cnn_n2v-pt.h5
```
