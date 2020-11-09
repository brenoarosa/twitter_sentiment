import os
import tempfile
import subprocess
from typing import Iterable, Tuple
from collections import namedtuple
import joblib
import numpy as np
import pandas as pd
import networkx as nx
from graph_tool import Graph
from graph_tool.spectral import adjacency
from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec
from sklearn.manifold import LocallyLinearEmbedding
from twitter_sentiment.utils import get_logger
from twitter_sentiment.graph.utils import load_graph, load_stellar_graph

logger = get_logger()

GraphEmbedding = namedtuple('GraphEmbedding', ["idx2user", "user2idx", "weights"])

ORDERED_USERS_FILEPATH = "data/output/graph_users_ordered.csv" #FIXME: hardcoded

def graph_embedding(edgelist_filepath: str, algorithm: str, output_path: str, prune_scc: bool = True) -> GraphEmbedding:

    if algorithm == "node2vec":
        embedding = node2vec(edgelist_filepath, prune_scc=prune_scc)

    elif algorithm == "node2vec_snap":
        embedding = node2vec(edgelist_filepath, "snap", prune_scc=prune_scc)

    elif algorithm == "node2vec_stellar":
        embedding = node2vec(edgelist_filepath, "stellar", prune_scc=prune_scc)

    elif algorithm == "lle":
        embedding = lle(edgelist_filepath, prune_scc)

    elif algorithm == "gcn":
        embedding = gcn(edgelist_filepath, prune_scc)

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

    if g.num_vertices() > MAX_VERTICES:
        logger.warning(f"Too many vertices, selecting [{MAX_VERTICES}] of them.")

        graph_users = pd.read_csv(ORDERED_USERS_FILEPATH, dtype={"user": str})
        graph_users = graph_users.loc[graph_users.in_scc == True] # filter scc
        graph_users = graph_users[0:MAX_VERTICES] # filter with most tweets
        graph_users = set(graph_users.user.to_list())

        indexes = []
        for i, user_id in enumerate(g.vp["user_ids"]):
            if user_id in graph_users:
                indexes.append(i)

        logger.warning(f"Filled [{len(indexes)}] vertices.")

        indexes = np.array(indexes)
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


def node2vec(edgelist_filepath: str, implementation: str = "snap", prune_scc: bool = True, **kwargs) -> GraphEmbedding:

    # default params
    params = {
        "dimensions": 128,
        "walk_length": 80,
        "walks_per_source": 10,
        "p": 1,
        "q": 1,
        "context_size": 10,
        "epochs": 3,
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

    MAX_VERTICES = 16000

    graph_users = pd.read_csv(ORDERED_USERS_FILEPATH, dtype={"user": str})
    graph_users = graph_users.loc[graph_users.in_scc == True] # filter scc
    graph_users = graph_users[0:MAX_VERTICES] # filter with most tweets
    graph_users = set(graph_users.user.to_list())

    c = 0
    with os.fdopen(edge_list_fd, 'w') as edgelist:
        for i, edge in enumerate(g.edges()):
            source_id = g.vp["user_ids"][int(edge.source())]
            target_id = g.vp["user_ids"][int(edge.target())]

            if (source_id in graph_users) and (target_id in graph_users):
                edgelist.write(f"{int(edge.source())} {int(edge.target())}\n")
                c += 1

    logger.info(f"Total edges written {c}")

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


def gcn(edgelist_filepath: str, prune_scc: bool = True, **kwargs) -> GraphEmbedding:
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    import tensorflow as tf
    from tensorflow.keras import Model
    from stellargraph.mapper import (
        CorruptedGenerator,
        FullBatchNodeGenerator,
        GraphSAGENodeGenerator,
        HinSAGENodeGenerator,
        ClusterNodeGenerator,
    )
    from stellargraph import StellarGraph
    from stellargraph.layer import GCN, DeepGraphInfomax, GraphSAGE, GAT, APPNP, HinSAGE

    params = {
        "dimensions": 128,
    }

    graph_users = pd.read_csv(ORDERED_USERS_FILEPATH, dtype={"user": str})
    graph_users = graph_users.loc[graph_users.in_scc == True] # filter scc
    graph_users = set(graph_users.user.to_list())

    df = pd.read_csv(edgelist_filepath, sep=",", dtype="str")
    df = df.loc[df.user_id.isin(graph_users) & df.retweeted_user_id.isin(graph_users)]
    nx_g = nx.from_pandas_edgelist(df, "user_id", "retweeted_user_id", create_using=nx.Graph)

    nx_g.remove_edges_from(nx.selfloop_edges(nx_g))
    if prune_scc:
        nx_g = nx_g.subgraph(max(nx.connected_components(nx_g), key=len))

    feat = pd.DataFrame.from_dict(dict(nx_g.degree()), orient="index", columns=["degree"])
    feat = np.log(1 + feat.degree)

    G = StellarGraph.from_networkx(nx_g, node_features=feat)

    fullbatch_generator = FullBatchNodeGenerator(G, sparse=True)
    gcn_model = GCN(layer_sizes=[params["dimensions"]], activations=["relu"], generator=fullbatch_generator)

    corrupted_generator = CorruptedGenerator(fullbatch_generator)
    gen = corrupted_generator.flow(G.nodes())

    infomax = DeepGraphInfomax(gcn_model, corrupted_generator)
    x_in, x_out = infomax.in_out_tensors()

    model = Model(inputs=x_in, outputs=x_out)
    model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer=Adam(lr=1e-3))

    epochs = 500
    es = EarlyStopping(monitor="loss", min_delta=0, patience=20)
    history = model.fit(gen, epochs=epochs, verbose=1, callbacks=[es])

    x_emb_in, x_emb_out = gcn_model.in_out_tensors()

    # for full batch models, squeeze out the batch dim (which is 1)
    x_out = tf.squeeze(x_emb_out, axis=0)
    emb_model = Model(inputs=x_emb_in, outputs=x_out)

    weights = emb_model.predict(fullbatch_generator.flow(G.nodes()))

    idx2user = {i: v for i, v in enumerate(G.nodes())}
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
