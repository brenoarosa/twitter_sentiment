import json
from graph_tool import Graph
from graph_tool.topology import label_largest_component
from graph_tool.clustering import global_clustering, local_clustering
from graph_tool.stats import distance_histogram
import numpy as np
from twitter_sentiment.graph.utils import load_graph
from twitter_sentiment.utils import get_logger


logger = get_logger()

def calculate_base_stats(g: Graph) -> dict:
    # graph structure
    logger.info("Collecting structure info...")
    n_edges = g.num_edges()
    n_vertices = g.num_vertices()

    retweets = g.ep['retweets'].fa
    retweets_stats = (float(retweets.mean()), float(retweets.std()), int(retweets.min()), int(retweets.max()))

    density = n_edges / (n_vertices * (n_vertices - 1))

    # largest component
    logger.info("Calculating largest component...")
    scc_filter = label_largest_component(g, directed=True)
    scc_vertice_percentage = float(scc_filter.fa.mean())

    wcc_filter = label_largest_component(g, directed=False)
    wcc_vertice_percentage = float(wcc_filter.fa.mean())

    # degree
    logger.info("Calculating degree stats...")
    k_in = g.get_in_degrees(g.get_vertices())
    k_out = g.get_out_degrees(g.get_vertices())
    k_total = k_in + k_out

    k_in_stats = get_stats(k_in)
    k_out_stats = get_stats(k_out)

    # clustering
    logger.info("Calculating global clustering stats...")
    global_c = global_clustering(g)[0]

    logger.info("Calculating local undirected clustering stats...")
    local_undirected_c = local_clustering(g, undirected=True)
    local_undirected_c_stats = float(local_undirected_c.fa.mean()), float(local_undirected_c.fa.std()), float(local_undirected_c.fa.min()), float(local_undirected_c.fa.max())

    logger.info("Calculating local directed clustering stats...")
    local_directed_c = local_clustering(g, undirected=False)
    local_directed_c_stats = float(local_directed_c.fa.mean()), float(local_directed_c.fa.std()), float(local_directed_c.fa.min()), float(local_directed_c.fa.max())

    logger.info("Calculating distances...")
    # Always samples, even with samples=None!
    # to test it try running 2x the same hist
    dist_samples = int(min(50000, n_vertices*0.05))
    dist_hist = distance_histogram(g, samples=dist_samples, float_count=False)
    dist_count, dist_value = dist_hist[0], dist_hist[1][:-1]

    """
    # Ground Truth
    distances = Counter()
    for vertex in g.vertices():
        v_distance = shortest_distance(g, source=vertex, directed=True)
        v_dist_count = np.unique(v_distance.a, return_counts=True)
        distances.update(dict(zip(v_dist_count[0], v_dist_count[1])))
    """

    dist_max = np.max(dist_value)
    dist_mean = (dist_count * dist_value).sum() / dist_count.sum()
    std_sum_squared = (dist_value - dist_mean)**2
    dist_std = np.sqrt((std_sum_squared * dist_count).sum() / (dist_count.sum() - 1))
    dist_stats = {
        "max": float(dist_max),
        "mean": float(dist_mean),
        "std": float(dist_std),
    }

    dist_dict = dict(zip(dist_count.tolist(), dist_value.tolist()))

    results = {
        "n_edges": n_edges,
        "n_vertices": n_vertices,
        "retweets_stats": retweets_stats,
        "density": density,
        "scc_vertice_percentage": scc_vertice_percentage,
        "wcc_vertice_percentage": wcc_vertice_percentage,
        "k_in_stats": k_in_stats,
        "k_out_stats": k_out_stats,
        "global_c": global_c,
        "local_undirected_c_stats": local_undirected_c_stats,
        "local_directed_c_stats": local_directed_c_stats,
        "dist_stats": dist_stats,
        "dist_dict": dist_dict
    }
    return results


def get_stats(values: np.ndarray):
    stats = {}
    stats["min"] = float(values.min())
    stats["max"] = float(values.max())
    stats["mean"] = float(values.mean())
    stats["std"] = float(values.std())
    stats["p25"] = float(np.quantile(values, 0.25))
    stats["p50"] = float(np.quantile(values, 0.50))
    stats["p75"] = float(np.quantile(values, 0.75))
    stats["p90"] = float(np.quantile(values, 0.90))
    stats["p95"] = float(np.quantile(values, 0.95))
    stats["p99"] = float(np.quantile(values, 0.99))
    return stats

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to edge list file", type=str)
    parser.add_argument("-o", "--output", help="Path to output stats file", type=str, required=True)
    args = parser.parse_args()
    logger.info("Loading graph...")
    g = load_graph(args.input)
    stats = calculate_base_stats(g)
    with open(args.output, 'w') as fout:
        json.dump(stats, fout, indent=4)
