import json
from graph_tool import Graph
from graph_tool.topology import label_largest_component
from graph_tool.clustering import global_clustering, local_clustering
from graph_tool.stats import distance_histogram
import numpy as np
from twitter_sentiment.graph.utils import load_graph


def calculate_base_stats(g: Graph) -> dict:
    # graph structure
    print("Collecting structure info...")
    n_edges = g.num_edges()
    n_vertices = g.num_vertices()

    retweets = g.ep['retweets'].fa
    retweets_stats = (float(retweets.mean()), float(retweets.std()), int(retweets.min()), int(retweets.max()))

    density = n_edges / (n_vertices * (n_vertices - 1))

    # largest component
    print("Calculating largest component...")
    scc_filter = label_largest_component(g, directed=True)
    scc_vertice_percentage = float(scc_filter.fa.mean())

    wcc_filter = label_largest_component(g, directed=False)
    wcc_vertice_percentage = float(wcc_filter.fa.mean())

    # degree
    print("Calculating degree stats...")
    k_in = g.get_in_degrees(g.get_vertices())
    k_out = g.get_out_degrees(g.get_vertices())
    k_total = k_in + k_out

    k_in_stats = k_in.mean(), k_in.std(), k_in.min(), k_in.max()
    k_out_stats = k_out.mean(), k_out.std(), k_out.min(), k_out.max()

    # clustering
    print("Calculating clustering stats...")
    global_c = global_clustering(g)[0]

    local_undirected_c = local_clustering(g, undirected=True)
    local_undirected_c_stats = float(local_undirected_c.fa.mean()), float(local_undirected_c.fa.std()), float(local_undirected_c.fa.min()), float(local_undirected_c.fa.max())

    local_directed_c = local_clustering(g, undirected=False)
    local_directed_c_stats = float(local_directed_c.fa.mean()), float(local_directed_c.fa.std()), float(local_directed_c.fa.min()), float(local_directed_c.fa.max())

    print("Calculating distances...")
    # Always samples, even with samples=None!
    # to test it try running 2x the same hist
    dist_hist = distance_histogram(g, samples=n_vertices*0.05, float_count=False)
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
    dist_stats = (dist_mean, dist_std, dist_max)

    dist_dict = dict(zip(dist_count.tolist(), dist_value.tolist()))

    results = {
        "n_edges": n_edges,
        "n_vertices": n_vertices,
        "retweets_stats": retweets_stats,
        "density": density,
        "scc_vertice_percentage": scc_vertice_percentage,
        "wcc_vertice_percentage": wcc_vertice_percentage,
        "k_in_stats": [float(k) for k in k_in_stats],
        "k_out_stats": [float(k) for k in k_out_stats],
        "global_c": global_c,
        "local_undirected_c_stats": local_undirected_c_stats,
        "local_directed_c_stats": local_directed_c_stats,
        "dist_stats": [float(stat) for stat in dist_stats],
        "dist_dict": dist_dict
    }
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to edge list file", type=str)
    parser.add_argument("-o", "--output", help="Path to output stats file", type=str, required=True)
    args = parser.parse_args()
    g = load_graph(args.input)
    stats = calculate_base_stats(g)
    with open(args.output, 'w') as fout:
        json.dump(stats, fout, indent=4)
