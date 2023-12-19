import networkx as nx
from networkx.drawing.nx_agraph import write_dot
import numpy as np
import graphviz
MG = nx.MultiGraph()
def get_edges_from_data(edata: list) -> list:
    res = []
    for i in edata:
        res.extend([i[0:2]] * i[2])
    return res
nodes = ['a', 'A', 'b', 'B', 'c', 'C']
sample_edata = [('a', 'B', 1), ('a', 'C', 1), ('b', 'A', 1), ('b', 'C', 1), ('c', 'A', 1), ('c', 'B', 1), ('c', 'C', 2)]
l = get_edges_from_data(sample_edata)
MG.add_nodes_from(nodes)
MG.add_edges_from(l)
write_dot(MG, 'multi.dot')