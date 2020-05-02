# Twitter Sentiment Classifier

Code implementing my [master thesis](https://github.com/brenoarosa/thesis).

## Dataset Pre-Processing
Extracting core tweet fields from raw tweets:
```sh
python -m twitter_sentiment.preprocessors.dataset_preprocess data/tweets/amazonia_lzma/*.jsonlines.lzma -o /tmp/amazonia-pt.jsonline.xz -l pt
```

Building edgelist from raw tweets:
```sh
python -m twitter_sentiment.graph.preprocess data/tweets/amazonia_lzma/*.jsonlines.lzma -o data/output/amazonia-edgelist.csv
```

Calculate base graph stats:
```sh
python -m twitter_sentiment.graph.stats data/output/amazonia-edgelist.csv -o data/output/amazonia-graph-stats.json
```

## Text Classification


## Graph Classification
