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
def get_edge_pairing_of_single_word(word, w_index):
    dict_of_edge_pairing = {"a": [], "b": [], "c": []}
    l = get_length_two_subwords([word])
    for i in range(len(l)):
        if i == 0:
            if l[0][0].islower():
                dict_of_edge_pairing[l[0][0]].append((len(l) - 1 + w_index, w_index))
            else:
                m = l[0][0].lower()
                if (w_index, len(l) - 1 + w_index) not in dict_of_edge_pairing[m]:
                    dict_of_edge_pairing[m].append((w_index, len(l) - 1 + w_index))
        elif l[i][0].islower():
            if (i-1 + w_index, i + w_index) not in dict_of_edge_pairing[l[i][0]]:
                dict_of_edge_pairing[l[i][0]].append((i-1 + w_index, i + w_index))
        else:
            m = l[0][0].lower()
            if (i + w_index, i-1 + w_index) not in dict_of_edge_pairing[m]:
                dict_of_edge_pairing[m].append((i + w_index, i-1 + w_index))             
    return dict_of_edge_pairing
def get_whitehead_graph_with_edge_pairing(list_of_words):
    # Rows: a, A, b, B, c, C
    dict_of_edge_pairing = {"a": [], "b": [], "c": []}
    for i in range(len(list_of_words)):
        d = get_edge_pairing_of_single_word(list_of_words[i], sum([len(list_of_words[j]) for j in range(i)], 0))
        dict_of_edge_pairing["a"].extend(d["a"])
        dict_of_edge_pairing["b"].extend(d["b"])
        dict_of_edge_pairing["c"].extend(d["c"])
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
    return (res, dict_of_edge_pairing)
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
    A_G = get_whitehead_graph_with_edge_pairing(list_of_words)[0]
    return is_connected(A_G) and is_pairwise_well_connected(A_G)
def get_all_cycles(incidence_matrix):
    all_c = []
    for v in itertools.product({0, 1}, repeat=len(incidence_matrix[0])):
        resulting_check = np.matmul(incidence_matrix, v)
        if all([i in {0, 2} for i in resulting_check]):
            all_c.append(v)
    all_c.pop(0)
    return all_c
A_G, edge_pairing = get_whitehead_graph_with_edge_pairing(sample_list_of_words)
# print(is_minimal_and_diskbusting(sample_list_of_words))
cycles = get_all_cycles(A_G)
cycle_mat = np.array(cycles)
# gurobi model
model = gp.Model("Find balanced list of cycles")
vars = model.addMVar(shape = (1, len(cycles)), vtype = GRB.INTEGER, lb = 0)
model.addConstr(gp.quicksum(vars) >= 1)
# conditions for balance
for gen in ["a", "b", "c"]:
    for (i, j) in itertools.combinations(edge_pairing[gen], 2):
        model.addConstr(gp.quicksum(vars[0, k] for k in range(len(cycles)) if cycles[k][i[0]] == 1 and cycles[k][j[0]] == 1) == gp.quicksum(vars[0, k] for k in range(len(cycles)) if cycles[k][i[1]] == 1 and cycles[k][j[1]] == 1))
# calculations
model.setObjective(gp.quicksum((vars @ cycle_mat)[0]), sense = GRB.MINIMIZE)
model.Params.MIPFocus = 1
model.optimize()
print(model.X)
# Assumption: each word is a power of order at least four of a certain word.
# Only condition (d) of (4.23) (Cyclic polytope) is possible.


