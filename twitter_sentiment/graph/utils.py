import pandas as pd
from graph_tool import Graph, GraphView
from graph_tool.topology import label_largest_component
from graph_tool.stats import remove_self_loops
import networkx as nx
import stellargraph as sg

def load_graph(filepath: str, trim_self_loop: bool = True, prune_scc: bool = False) -> Graph:
    df = pd.read_csv(filepath, sep=",", dtype="str")
    df = df.groupby(["user_id", "retweeted_user_id"]).size().reset_index(name='retweet_count')
    g = Graph()

    def filter_scc(g: Graph) -> Graph:
        g = GraphView(g, vfilt=label_largest_component(g, directed=True))
        return Graph(g, prune=True)

    def get_df_iterator(df):
        for thrice in df.to_records(index=False):
            source, target, count = thrice
            yield([source, target, count])

    retweets_prop = g.new_edge_property("int")
    eprops = [retweets_prop]
    g.add_edge_list(get_df_iterator(df), hashed=True, string_vals=True, eprops=eprops)
    g.edge_properties["retweets"] = retweets_prop

    if trim_self_loop:
        remove_self_loops(g)

    if prune_scc:
        g = filter_scc(g)
    return g

def load_stellar_graph(filepath: str, trim_self_loop: bool = True, prune_scc: bool = False) -> sg.StellarGraph:
    df = pd.read_csv(filepath, sep=",", dtype="str")
    df = df.groupby(["user_id", "retweeted_user_id"]).size().reset_index(name='retweet_count')

    nx_g = nx.from_pandas_edgelist(df, "user_id", "retweeted_user_id", edge_attr="retweet_count", create_using=nx.DiGraph)

    if trim_self_loop:
        nx_g.remove_edges_from(nx.selfloop_edges(nx_g))

    if prune_scc:
        if nx_g.is_directed():
            nx_g = nx_g.subgraph(max(nx.strongly_connected_components(nx_g), key=len))
        else:
            nx_g = nx_g.subgraph(max(nx.connected_components(nx_g), key=len))

    g = sg.StellarGraph.from_networkx(nx_g)
    return g
