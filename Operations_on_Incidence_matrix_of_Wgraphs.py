import gurobipy as gp
from gurobipy import GRB
import numpy as np
import itertools
# import igraph
import networkx
# We work over the free group of rank 3.
generator = ["a", "b", "c", "A", "B", "C"]
n = len(generator)
sample_list_of_words = ["aB","bC", "ca", "Ab", "Bc"]
def get_length_two_subwords(list_of_words):
    list_of_length_two_subwords = []
    for i in list_of_words:
        for j in range(-1, len(i)-1):
            list_of_length_two_subwords.append(i[j] + i[j+1])
    return list_of_length_two_subwords
# print(get_length_two_subwords(sample_list_of_words))
gen_dict = {"a": 0, "A": 1, "b": 2, "B": 3, "c": 4, "C": 5}
def get_whitehead_graph(list_of_words):
    # Rows: a, A, b, B, c, C
    l = get_length_two_subwords(list_of_words)
    res = np.zeros((n, len(l)))
    for i in range(len(l)):
        w = list(l[i])
        res[gen_dict[w[0]], i] = 1
        if w[1].islower():
            w[1] = w[1].upper()
            res[gen_dict[w[1]], i] = 1
        else:
            w[1] = w[1].lower()
            res[gen_dict[w[1]], i] = 1
    return res
def is_connected(whitehead_graph):
    return np.linalg.matrix_rank(whitehead_graph) == n - 1
def is_pairwise_well_connected(whitehead_graph):
    adj_matrix = np.matmul(whitehead_graph, whitehead_graph.T)
    np.fill_diagonal(adj_matrix, 0)
    G = networkx.from_numpy_array(adj_matrix)
    w = networkx.get_edge_attributes(G, "weight")
    networkx.set_edge_attributes(G, w, "capacity")
    degs = G.degree()
    for i in range(int(n/2)):
        flow_value = networkx.maximum_flow_value(G, 2 * i, 2 * i + 1)
        if flow_value != degs[2 * i]:    
            return False
    return True
def is_minimal_and_diskbusting(list_of_words):
    A_G = get_whitehead_graph(list_of_words)
    return is_connected(A_G) and is_pairwise_well_connected(A_G)
def get_all_cycles(incidence_matrix):
    all_c = []
    for v in itertools.product({0, 1}, repeat=len(incidence_matrix[0])):
        resulting_check = np.matmul(incidence_matrix, v)
        if all([i in {0, 2} for i in resulting_check]):
            all_c.append(v)
    return all_c
A_G = get_whitehead_graph(sample_list_of_words)
print(is_minimal_and_diskbusting(sample_list_of_words))
print(get_all_cycles(A_G))
# Assumption: each word is a power of order at least four of a certain word.
# Only condition (d) of (4.23) is possible.


