from graph_tool import Graph, GraphView
from graph_tool.topology import label_largest_component
from graph_tool.stats import remove_self_loops
import pandas as pd

def load_graph(filepath: str) -> Graph:
    df = pd.read_csv(filepath, sep=",", dtype="str")
    df = df.groupby(["user_id", "retweeted_user_id"]).size().reset_index(name='retweet_count')
    g = Graph()

    def get_df_iterator(df):
        for thrice in df.to_records(index=False):
            source, target, count = thrice
            yield([source, target, count])

    retweets_prop = g.new_edge_property("int")
    eprops = [retweets_prop]
    g.add_edge_list(get_df_iterator(df), hashed=True, string_vals=True, eprops=eprops)
    g.edge_properties["retweets"] = retweets_prop

    remove_self_loops(g)
    return g

def filter_scc(g: Graph) -> Graph:
    g = GraphView(g, vfilt=label_largest_component(g, directed=True))
    return Graph(g, prune=True)
