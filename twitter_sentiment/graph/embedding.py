import os
import tempfile
import subprocess
from graph_tool import Graph
from twitter_sentiment.utils import get_logger
from twitter_sentiment.graph.utils import load_graph, filter_scc


logger = get_logger()

def graph_embedding(g: Graph, algorithm: str, prune_scc: bool = True) -> None:
    if prune_scc:
        g = filter_scc(g)

    if algorithm == "node2vec":
        node2vec(g)
    else:
        raise ValueError("Invalid embedding algorithm")

def node2vec(g: Graph):
    edge_list_fd, edge_list_filepath = tempfile.mkstemp()
    logger.info("Writing node2vec edgelist to [%s]", edge_list_filepath)
    with os.fdopen(edge_list_fd, 'w') as edgelist:
        for edge in g.edges():
            edgelist.write(f"{int(edge.source())} {int(edge.target())}\n")

    _, embedding_filepath = tempfile.mkstemp()

    cmd = [os.getenv("NODE2VEC_PATH"), f"-i:{edge_list_filepath}", f"-o:{embedding_filepath}", "-d:128", "-l:80", "-r:10", "-k:10", "-e:1", "-p:1", "-q:1", "-v"]
    logger.info("Running [%s]", " ".join(cmd))
    logger.info("Writing embedding file to [%s]", embedding_filepath)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to edge list file", type=str)
    #parser.add_argument("-o", "--output", help="Path to output file", type=str, required=True)
    args = parser.parse_args()

    logger.info("Loading graph [%s]", args.input)
    g = load_graph(args.input)
    graph_embedding(g, "node2vec")
