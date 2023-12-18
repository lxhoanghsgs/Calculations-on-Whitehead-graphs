import time
import gurobipy as gp
import math
from gurobipy import GRB
import numpy as np
import itertools
# import igraph
import networkx
import networkx.algorithms.isomorphism as iso
def stars_and_bars(stars: int, bars: int) -> list:
    """
    Euler division with "stars" candies and "bars" children.
    """
    for c in itertools.combinations(range(stars + bars - 1), bars - 1):
        yield [b-a-1 for a, b in zip((-1,)+c, c+(stars + bars -1,))]

generator = ["a", "b", "c", "A", "B", "C"]
generator_for_nx_nodes = [(i, {"name": i}) for i in generator]
all_edges = list(itertools.combinations(generator, 2))
e = len(all_edges)
n = len(generator)
gen_dict = {"a": 0, "A": 1, "b": 2, "B": 3, "c": 4, "C": 5}
reversed_gen_dict = {0: 'a', 1: 'A', 2: 'b', 3: 'B', 4: 'c', 5: 'C'}
def is_connected_and_pairwise_well_connected(wgraph: networkx.Graph) -> bool:
    if not networkx.is_connected(wgraph):
        return False
    w = networkx.get_edge_attributes(wgraph, "weight")
    networkx.set_edge_attributes(wgraph, w, "capacity")
    degs = wgraph.degree(weight= "weight")
    for i in generator[0:3]:
        j = i.upper()
        flow_value = networkx.maximum_flow_value(wgraph, i, j)
        if flow_value != degs[i] or flow_value != degs[j]:
            return False
    return True 
def get_wgraphs_with_number_of_edges(n_edge: int) -> list:
    """
    Get all connected and pairwise well-connected Whitehead graphs on six vertices.
    """
    res = []
    for edge_attr in stars_and_bars(n_edge, e):
        empty_graph = networkx.Graph()
        empty_graph.add_nodes_from(generator_for_nx_nodes)
        edge_bunch = [all_edges[i] + ({"weight": edge_attr[i]},) for i in range(e) if edge_attr[i] > 0]
        empty_graph.add_edges_from(edge_bunch)
        if is_connected_and_pairwise_well_connected(empty_graph):
            res.append(empty_graph)
    return res
n_edge = 8
t1 = time.perf_counter()
for k in range(9, 12):
    n_edge = k
    with open(f"whitehead_graphs_with_{n_edge}_edges.txt", 'w') as f:
        g_data = get_wgraphs_with_number_of_edges(n_edge)
        for i in range(len(g_data)):
            print(f"Wgraph {i} \n", file = f)
            print(f"{g_data[i].edges(data = 'weight')}\n", file = f)
        f.close()
t2 = time.perf_counter()
print(f"time_run = {round(t2 - t1, 2)}")
