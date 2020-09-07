import os
import tempfile
import subprocess
from typing import Iterable, Tuple
from collections import namedtuple
import joblib
import numpy as np
import pandas as pd
from graph_tool import Graph
from graph_tool.spectral import adjacency
from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec
from sklearn.manifold import LocallyLinearEmbedding
from twitter_sentiment.utils import get_logger
from twitter_sentiment.graph.utils import load_graph, load_stellar_graph


logger = get_logger()

GraphEmbedding = namedtuple('GraphEmbedding', ["idx2user", "user2idx", "weights"])

def graph_embedding(edgelist_filepath: str, algorithm: str, output_path: str, prune_scc: bool = True) -> GraphEmbedding:

    gcn(edgelist_filepath, prune_scc)

    if algorithm == "node2vec":
        embedding = node2vec(edgelist_filepath, prune_scc=prune_scc)

    elif algorithm == "node2vec_snap":
        embedding = node2vec(edgelist_filepath, "snap", prune_scc=prune_scc)

    elif algorithm == "node2vec_stellar":
        embedding = node2vec(edgelist_filepath, "stellar", prune_scc=prune_scc)

    elif algorithm == "lle":
        embedding = lle(edgelist_filepath, prune_scc)

    else:
        raise ValueError("Invalid embedding algorithm")

    joblib.dump(embedding, output_path)
    return embedding

def lle(edgelist_filepath: str, prune_scc: bool = True, **kwargs) -> GraphEmbedding:
    g = load_graph(edgelist_filepath, prune_scc=prune_scc)
    A = adjacency(g)

    MAX_VERTICES = 10000

    params = {
        "dimensions": 128,
    }

    n_vertices = g.num_vertices()
    indexes = np.arange(n_vertices)

    if n_vertices > MAX_VERTICES:
        logger.warning(f"Too many vertices, sampling [{MAX_VERTICES}] of them.")

        indexes = np.random.choice(n_vertices, MAX_VERTICES, replace=False)
        indexes.sort()
        A = A[indexes, :][:, indexes]

    logger.info("Calculating embedding...")
    embedding = LocallyLinearEmbedding(n_components=params["dimensions"], n_jobs=-1)
    vertice_embedding = embedding.fit_transform(A.todense())

    weights = vertice_embedding
    idx2user = dict(enumerate(indexes))
    user2idx = {v: k for k, v in enumerate(indexes)}
    emb = GraphEmbedding(idx2user=idx2user, user2idx=user2idx, weights=weights)
    return emb

def gcn(edgelist_filepath: str, prune_scc: bool = True, **kwargs) -> GraphEmbedding:

    import stellargraph as sg
    from stellargraph.data import EdgeSplitter
    from stellargraph.mapper import FullBatchLinkGenerator
    from stellargraph.layer import GCN, LinkEmbedding


    from tensorflow import keras
    from sklearn import preprocessing, feature_extraction, model_selection

    G = load_stellar_graph(edgelist_filepath, prune_scc=prune_scc)
    import ipdb; ipdb.set_trace()

    # Define an edge splitter on the original graph G:
    edge_splitter_test = EdgeSplitter(G)

    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G, and obtain the
    # reduced graph G_test with the sampled links removed:
    G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
        p=0.1, method="global", keep_connected=True
    )

    # Define an edge splitter on the reduced graph G_test:
    edge_splitter_train = EdgeSplitter(G_test)

    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G_test, and obtain the
    # reduced graph G_train with the sampled links removed:
    G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
        p=0.1, method="global", keep_connected=True
    )

    epochs = 50
    train_gen = FullBatchLinkGenerator(G_train, method="gcn")
    train_flow = train_gen.flow(edge_ids_train, edge_labels_train)

    test_gen = FullBatchLinkGenerator(G_test, method="gcn")
    test_flow = train_gen.flow(edge_ids_test, edge_labels_test)

    gcn = GCN(
        layer_sizes=[16, 16], activations=["relu", "relu"], generator=train_gen, dropout=0.3
    )

    x_inp, x_out = gcn.in_out_tensors()

    prediction = LinkEmbedding(activation="relu", method="ip")(x_out)

    prediction = keras.layers.Reshape((-1,))(prediction)

    model = keras.Model(inputs=x_inp, outputs=prediction)

    model.compile(
        optimizer=keras.optimizers.Adam(lr=0.01),
        loss=keras.losses.binary_crossentropy,
        metrics=["acc"],
    )

    init_train_metrics = model.evaluate(train_flow)
    init_test_metrics = model.evaluate(test_flow)

    print("\nTrain Set Metrics of the initial (untrained) model:")
    for name, val in zip(model.metrics_names, init_train_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    print("\nTest Set Metrics of the initial (untrained) model:")
    for name, val in zip(model.metrics_names, init_test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    history = model.fit(
        train_flow, epochs=epochs, validation_data=test_flow, verbose=2, shuffle=False
    )

    train_metrics = model.evaluate(train_flow)
    test_metrics = model.evaluate(test_flow)

    print("\nTrain Set Metrics of the trained model:")
    for name, val in zip(model.metrics_names, train_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    print("\nTest Set Metrics of the trained model:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))



def node2vec(edgelist_filepath: str, implementation: str = "snap", prune_scc: bool = True, **kwargs) -> GraphEmbedding:

    # default params
    params = {
        "dimensions": 128,
        "walk_length": 80,
        "walks_per_source": 10,
        "p": 1,
        "q": 1,
        "context_size": 10,
        "epochs": 1,
        "directed": False,
        "weighted": False,
    }

    # update params
    for k, v in kwargs.items():
        if k in params.keys():
            params[k] = v

    if implementation == "snap":
        g = load_graph(edgelist_filepath, prune_scc=prune_scc)
        return node2vec_snap(g, params=params)

    elif implementation == "stellar":
        g = load_stellar_graph(edgelist_filepath, prune_scc=prune_scc)
        rw = BiasedRandomWalk(g)

        # warning: python implementation and not parallelized, only works for really small graphs
        # FIXME: could use snap biased random walks (snap/examples/node2vec/node2vec -i:tmp5x7mq80f -o:out.txt -dr -v -ow)
        walks = rw.run(
            nodes=list(g.nodes()),  # root nodes
            length=80,  # maximum length of a random walk
            n=10,  # number of random walks per root node
            p=1,  # Defines (unormalised) probability, 1/p, of returning to source node
            q=1,  # Defines (unormalised) probability, 1/q, for moving away from source node
        )

        str_walks = [[str(n) for n in walk] for walk in walks]
        model = Word2Vec(str_walks, size=128, window=5, min_count=0, sg=1, workers=4, iter=1)

        weights = model.wv.vectors
        idx2user = dict(enumerate(model.wv.index2entity))
        user2idx = {v: k for k, v in enumerate(model.wv.index2entity)}
        emb = GraphEmbedding(idx2user=idx2user, user2idx=user2idx, weights=weights)
        return emb

    else:
        raise RuntimeError("Invalid argument implementation value.")


def node2vec_snap(g: Graph, params) -> GraphEmbedding:
    edge_list_fd, edge_list_filepath = tempfile.mkstemp()
    logger.info("Writing node2vec edgelist to [%s]", edge_list_filepath)
    logger.info(f"Graph with [{g.num_edges()}] nodes and [{g.num_vertices()}] edges.")

    MAX_EDGES = 1000000
    sample_output = False

    if g.num_edges() > MAX_EDGES:
        logger.warning(f"Too many edges, sampling [{MAX_EDGES}] of them.")
        sample_output = True
        sample_rate = MAX_EDGES / g.num_edges()
        samples = np.random.rand(g.num_edges())

    with os.fdopen(edge_list_fd, 'w') as edgelist:
        for i, edge in enumerate(g.edges()):
            if sample_output and (samples[i] > sample_rate):
                continue

            edgelist.write(f"{int(edge.source())} {int(edge.target())}\n")

    _, embedding_filepath = tempfile.mkstemp()

    # param strings
    dimensions = "-d:" + str(params["dimensions"])
    walk_length = "-l:" + str(params["walk_length"])
    walks_per_source = "-r:" + str(params["walks_per_source"])
    context_size = "-k:" + str(params["context_size"])
    epochs = "-e:" + str(params["epochs"])
    p = "-p:" + str(params["p"])
    q = "-q:" + str(params["q"])

    cmd = [os.environ["NODE2VEC_PATH"], f"-i:{edge_list_filepath}", f"-o:{embedding_filepath}",
           dimensions, walk_length, walks_per_source, context_size, epochs, p, q, "-v"]

    if params["directed"]:
        cmd.append("-dr")

    if params["weighted"]:
        cmd.append("-w")

    logger.info("Running [%s]", " ".join(cmd))
    logger.info("Writing embedding file to [%s]", embedding_filepath)
    subprocess.run(cmd, check=True)

    names = ["vertex_index"] + [f"emb_{i}" for i in range(params["dimensions"])]
    df = pd.read_csv(embedding_filepath, sep=" ", header=None, names=names, skiprows=1)

    user_ids = g.vp["user_ids"]
    df["user_id"] = df.vertex_index.apply(lambda x: user_ids[x])
    df = df.drop(columns="vertex_index")

    weights = df.drop(columns="user_id").values

    idx2user = df["user_id"].to_dict()
    user2idx = {v: k for k, v in idx2user.items()}

    emb = GraphEmbedding(idx2user=idx2user, user2idx=user2idx, weights=weights)
    return emb


def load_graph_emb_weight_and_X(model: GraphEmbedding, tweets: Iterable[dict]) -> Tuple[np.ndarray, np.ndarray]:

    tweets_users = [t["user_id"] for t in tweets] # generator to list

    X = np.zeros(len(tweets_users))

    for i, user in enumerate(tweets_users):
        user_idx = model.user2idx.get(user, 0)
        X[i] = user_idx

    vector_size = model.weights.shape[1]
    pad_vector = np.zeros(vector_size)
    weights = np.vstack((pad_vector, model.weights))
    return weights, X

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to edge list file", type=str)
    parser.add_argument("-o", "--output", help="Path to output file", type=str, required=True)
    parser.add_argument("-a", "--algorithm", help="Algorithm", type=str, default="node2vec")
    args = parser.parse_args()

    graph_embedding(args.input, algorithm=args.algorithm, output_path=args.output)
