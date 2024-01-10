import time
import gurobipy as gp
import math
from gurobipy import GRB
import numpy as np
import itertools
import networkx
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
def is_reg_for_wgraph(G: networkx.Graph) -> bool:
    """
    Returns True if G is regular as a multigraph.
    """
    degs = [0, 0, 0]
    for e in G.edges(data= 'weight'):
        if 'a' in e:
            degs[0] = degs[0] + e[-1]
        if 'b' in e:
            degs[1] = degs[1] + e[-1]
        if 'c' in e:
            degs[2] = degs[2] + e[-1]
    return degs[0] == degs[1] and degs[1] == degs[2]

def get_wgraphs_with_number_of_edges(n_edge: int) -> list:
    """
    Get all connected and pairwise well-connected Whitehead graphs on six vertices.
    """
    for edge_attr in stars_and_bars(n_edge, e):
        empty_graph = networkx.Graph()
        empty_graph.add_nodes_from(generator_for_nx_nodes)
        edge_bunch = [all_edges[i] + ({"weight": edge_attr[i]},) for i in range(e) if edge_attr[i] > 0]
        empty_graph.add_edges_from(edge_bunch)
        if is_connected_and_pairwise_well_connected(empty_graph) and not any([is_reg_for_wgraph(empty_graph), 'a' in empty_graph['A'], 'b' in empty_graph['B'], 'c' in empty_graph['C']]):
            yield empty_graph
n_edge = 8
t1 = time.perf_counter()
with open(f"Interesting_whitehead_graphs_with_{n_edge}_edges.txt", 'w') as f:
    count = 0
    for G in get_wgraphs_with_number_of_edges(n_edge):
        count += 1
        f.write(f"Wgraph number {count} \n")
        f.write(f"{G.edges(data = 'weight')}\n")
        print(count)
        if count == 3001:
            break
    f.close()
t2 = time.perf_counter()
print(f"time_run = {round(t2 - t1, 2)}")
# A small question: Are there any case that we can ensure that these graphs must have edges between corresponding vertices?!
# We can omit the case that the resulting graph is regular.