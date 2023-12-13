import gurobipy as gp
from gurobipy import GRB
import numpy as np
import itertools
# import igraph
import networkx
# We work over the free group of rank 3.
generator = ["a", "b", "c", "A", "B", "C"]
n = len(generator)
sample_list_of_words = ["abcABcc"]
def get_length_two_subwords(list_of_words):
    list_of_length_two_subwords = []
    for i in list_of_words:
        for j in range(-1, len(i)-1):
            list_of_length_two_subwords.append(i[j] + i[j+1])
    return list_of_length_two_subwords
# print(get_length_two_subwords(sample_list_of_words))
gen_dict = {"a": 0, "A": 1, "b": 2, "B": 3, "c": 4, "C": 5}
def is_cyclically_reduced(word: str) -> bool:
    return all(i not in get_length_two_subwords([word,]) for i in ["aA", "Aa", "bB", "Bb", "cC", "Cc"])
def get_inverse(word: str) -> str:
    inversed_word = list(word[len(word) - 1 - i] for i in range(len(word)))
    for i in range(len(inversed_word)):
        if inversed_word[i].islower():
            inversed_word[i] = inversed_word[i].upper()
        else:
            inversed_word[i] = inversed_word[i].lower()
    return "".join(inversed_word)
def get_redundant_words(word: str) -> set:
    l = len(word)
    A = set("".join([word[i - j] for i in range(l)]) for j in range(l))
    B = set(get_inverse(w) for w in A)
    return A.union(B)

def get_edge_pairing_of_single_word(word: str, w_index: int) -> dict:
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
def get_whitehead_graph_with_edge_pairing(list_of_words: list):
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
def is_connected(whitehead_graph: np.ndarray) -> bool:
    return np.linalg.matrix_rank(whitehead_graph) == n - 1
def is_pairwise_well_connected(whitehead_graph):
    adj_matrix = np.matmul(whitehead_graph, whitehead_graph.T)
    np.fill_diagonal(adj_matrix, 0)
    G = networkx.from_numpy_array(adj_matrix)
    w = networkx.get_edge_attributes(G, "weight")
    networkx.set_edge_attributes(G, w, "capacity")
    degs = G.degree(weight = "weight")
    for i in range(int(n/2)):
        flow_value = networkx.maximum_flow_value(G, 2 * i, 2 * i + 1)
        if flow_value != degs[2 * i]:    
            return False
    return True
def is_minimal_and_diskbusting(list_of_words: list) -> bool:
    A_G = get_whitehead_graph_with_edge_pairing(list_of_words)[0]
    return is_connected(A_G) and is_pairwise_well_connected(A_G)
def is_subset(cyc1: tuple, cyc2: tuple) -> bool:
    return all(cyc1[i] <= cyc2[i] for i in range(len(cyc1)))
def get_all_cycles(incidence_matrix: np.ndarray) -> list:
    all_c = []
    for v in itertools.product({0, 1}, repeat=len(incidence_matrix[0])):
        resulting_check = np.matmul(incidence_matrix, v)
        if all([i in {0, 2} for i in resulting_check]) and any(i > 0 for i in resulting_check):
            if any(is_subset(cyc1, v) for cyc1 in all_c):
                pass
            else:
                all_c.append(v)
    return all_c
A_G, edge_pairing = get_whitehead_graph_with_edge_pairing(sample_list_of_words)
print(is_minimal_and_diskbusting(sample_list_of_words))
cycles = get_all_cycles(A_G)
cycle_mat = np.array(cycles)
print(cycles)

# gurobi model
model = gp.Model("Find balanced list of cycles")
vars = model.addMVar(shape = (1, len(cycles)), vtype = GRB.INTEGER, lb = 0)
model.addConstr(gp.quicksum(vars) >= 1)
# conditions for balance
for gen in ["a", "b", "c"]:
    for (i, j) in itertools.combinations(edge_pairing[gen], 2):
        model.addConstr(gp.quicksum(vars[0, k] for k in range(len(cycles)) if cycles[k][i[0]] == 1 and cycles[k][j[0]] == 1) == gp.quicksum(vars[0, k] for k in range(len(cycles)) if cycles[k][i[1]] == 1 and cycles[k][j[1]] == 1))
# calculations
model.setObjective(0, sense = GRB.MINIMIZE)
model.Params.MIPFocus = 1
# focus on finding feasible solutions
model.optimize()

# sol = [int(i) for i in model.X]
# print(sol) model is wrong somewhere. I'll check later.

def get_set_of_all_good_words_of_length_at_most(length : int) -> set:
    # minimal_diskbusting_cyclically_reduced
    res = set()
    for i in range(1, length + 1):
        for j in itertools.product(generator, repeat = i):
            word = "".join(j)
            if is_cyclically_reduced(word) and is_minimal_and_diskbusting([word,]) and res.isdisjoint(get_redundant_words(word)):
                res.add(word)
    return res
# checklist = get_set_of_all_good_words_of_length_at_most(7)
# print(checklist)
"""
{'abbcAcB', 'abCacb', 'aabACbC', 'acABBBC', 'aBaCCbc', 'aBBCBaC', 'acBBcaB', 'aBCaCBB', 'aaacBBc', 'aabcbac', 'abaCbcc', 'aBBcABC', 'abcacbb', 'abbcaC', 'acACbbb', 'acBcaBB', 'abACbC', 'abbCaCB', 'acABBBc', 'abbcAC', 'aaCCaBB', 'abaCCbc', 'aaaBBCC', 'abacbbc', 'aBBcaC', 'aaacbCB', 'aaccaBB', 'acaBBC', 'aBBaCBC', 'aBaCBc', 'abbaccc', 'acaCbb', 'abcBBaC', 'aabCaBc', 'aabbaCC', 'aaBBCCB', 'abCCCaB', 'abaBcc', 'aabcBC', 'acaBBcb', 'aBaCBCC', 'aabbCCC', 'aaCCabb', 'acbAbCC', 'aaaCCBB', 'aaacbbc', 'abCAccb', 'aacbcAB', 'acbaCB', 'aacABCB', 'aabACBC', 'aaBBCCC', 'abCBaC', 'acACbb', 'acAcBBB', 'aaBCCBB', 'aBccACB', 'abacBC', 'abbbCac', 'abCBac', 'abCbbaC', 'aacbcab', 'aacbbbc', 'abACCCB', 'aBCaCbb', 'acAbbc', 'abCCbaC', 'acbbbAc', 'aBcbbaC', 'abccaCb', 'acbACbb', 'abccbac', 'acaBccB', 'abcbAc', 'aabcbAc', 'abAbcc', 'abABCCC', 'abbcBac', 'abaccbc', 'aBBCbaC', 'aaabccb', 'aBCbbaC', 'abcAbC', 'aaCCCBB', 'aaBCCB', 'aaaBCbc', 'acbAcB', 'aaBBccB', 'aBccAB', 'aaacBCB', 'abbbacc', 'aaccbbb', 'aaCCbbb', 'acABBC', 'aabCbAC', 'aaaCbCB', 'acAcBB', 'aaccBB', 'abAcbc', 'aacccbb', 'aaccBBB', 'abABcc', 'aaBCCCB', 'abbcacb', 'aBCCBaC', 'abaccBC', 'aacbCb', 'abaCbc', 'aacAbcb', 'aBaCCBc', 'abCBBac', 'aabcbAC', 'aBBBcAC', 'aaabCCb', 'abbCAc', 'acccbAb', 'aabbacc', 'acBcaB', 'aabaCbC', 'acBBaC', 'abACCb', 'aabAcbc', 'aaBccBB', 'aabccb', 'aaCCbbC', 'abABCC', 'aacbaCB', 'aacBcaB', 'acccbaB', 'abbCac', 'aaabcBC', 'aaBaCBC', 'acaBcb', 'aBBcAcb', 'aacbcB', 'aaaCbcB', 'aBBcbaC', 'aBcAcb', 'aabCCbb', 'aabbbcc', 'abacbC', 'acAbbbc', 'aaaccbb', 'abcaccb', 'aBcccAB', 'aaBCBAc', 'accBCaB', 'acBaCb', 'abbCAC', 'abccAB', 'aBaCCBC', 'aBaCBC', 'aBaCbcc', 'abaBCCC', 'aacBaBc', 'acBaBcc', 'abaCbbC', 'abbCaCb', 'aabCbaC', 'aaCbCAB', 'accbCaB', 'abbcBaC', 'acAbbC', 'acbbaBc', 'abccaB', 'abaCCbC', 'aBBaCC', 'abAcBc', 'aBcACCB', 'aaaBCBc', 'aBBcAC', 'abCAbc', 'aaCBacb', 'aaCBBC', 'aaCCBBB', 'aacBCB', 'abaccbC', 'abccACb', 'acAbCb', 'abCABCC', 'aaCAbcb', 'abcaCCb', 'abCCABC', 'acBccaB', 'aacbabc', 'aacBBcc', 'abcACCb', 'aabbcc', 'aBCaCb', 'abaBccc', 'aaCabCb', 'abCaCB', 'aBBcAc', 'aaabcbC', 'aaBCabc', 'aaCbCab', 'abcABc', 'aBCBaCC', 'abAccb', 'acBcAB', 'aabbbCC', 'aaCABCB', 'acbbAC', 'accaBB', 'aacaBcB', 'aaCbbbC', 'abcBBac', 'abCCCAb', 'aBcaCB', 'aacABcB', 'aaCaBCB', 'aacBaCb', 'aaCbcB', 'aaCBCb', 'abbbaCC', 'aaabCBc', 'aacbCB', 'acBCCaB', 'aaBacBc', 'aaaCbcb', 'abaCbCC', 'abaCCBc', 'aBccaCB', 'aabcaBC', 'aaBCBAC', 'aBccAbc', 'abcccaB', 'aBcAbc', 'aBaCCCb', 'abCaBc', 'abCbAc', 'abcAcBB', 'abCCAB', 'aaBcBAC', 'abaBCC', 'acaBBCb', 'acaBBcB', 'aaaCBcB', 'aaabbcc', 'aabCCb', 'aaaBcBC', 'abCACBB', 'aBCaCCB', 'aabCbc', 'acABCB', 'abAbccc', 'abcbac', 'acBCaB', 'aaaBcbC', 'abCCAb', 'aaaCBCb', 'abbacbc', 'aBBCaCB', 'aaCBcB', 'acbCCaB', 'acAcbbb', 'aabccbb', 'aaCbCAb', 'aabbccc', 'aacabcb', 'aaCBCAb', 'acbABBc', 'aBaCBcc', 'abCACB', 'abACCB', 'aaBcBC', 'aaaccBB', 'aaCABcB', 'abCbaCC', 'acBBaBc', 'acbCaB', 'abccAb', 'aaCBBCC', 'acbbaC', 'aabCaCb', 'abCCCAB', 'acbbABc', 'aaCBCAB', 'aaccabb', 'aaCBBBC', 'acbaBc', 'accBcaB', 'aBcABC', 'aaacbcB', 'acccbAB', 'aaBACbC', 'aBCBBaC', 'abCAcb', 'acbbAc', 'aaBcbC', 'abbcAc', 'abacbc', 'abacbCC', 'abbaCC', 'acbaBBc', 'abcAcB', 'abbbcAc', 'abaCCCB', 'abCBBaC', 'aaCbacB', 'abbcAbC', 'abcAbbC', 'abAbCCC', 'abaCBcc', 'abbbcAC', 'aaCBcb', 'abABccc', 'aaacBCb', 'abbbcaC', 'aaCCBB', 'aaBBBCC', 'abcbbac', 'abCaCbb', 'abCCacb', 'aaacbCb', 'acaBcBB', 'aaCBaBC', 'abbCACB', 'aBBaCCC', 'abAcccb', 'abbacc', 'accBaBC', 'aaBAcBc', 'aBCBaC', 'abcccAb', 'aaCAbCb', 'aaabcBc', 'aBBBcaC', 'accaBBB', 'accaBcB', 'acaCBBB', 'abbbCAC', 'aabcbC', 'aBcAbcc', 'aBBBaCC', 'abacccB', 'acACBB', 'aaBCbC', 'acccaBB', 'acBaBC', 'aacBcAB', 'aaBAcbc', 'aaCCbb', 'abcbacc', 'aaccBBc', 'aacAbCb', 'abcACb', 'aacbbcc', 'acbbbAC', 'aacccBB', 'abbcacB', 'abAbCC', 'abbbCAc', 'abacBCC', 'accbAbC', 'accbAcB', 'aaccbbc', 'aBCaCB', 'abCAbbc', 'acaBcbb', 'aBCbaC', 'aaaBccB', 'aaCbCB', 'acbAccB', 'abcaCb', 'accbAb', 'abCaCCb', 'acbACb', 'aaCbbC', 'acaBcB', 'aaBccB', 'aabbCCb', 'acbcAB', 'abCaccb', 'aaacBcb', 'acbcAb', 'aacBcb', 'aaabCBC', 'aBcABBC', 'abCCaB', 'aaCCCbb', 'aaCBCaB', 'aBcaCCB', 'aaaBcbc', 'aaBcBac', 'acBBBaC', 'aaaBCCB', 'abACCCb', 'aBBCaCb', 'abCbAC', 'aaBCBaC', 'abcaBC', 'aaBCaCB', 'abccacb', 'aaccbb', 'abcBac', 'acAbbbC', 'aBaCBBC', 'acaCBB', 'aaBACBC', 'aacBBBc', 'aaBBccc', 'acBaBBc', 'aaCbabC', 'aacbcAb', 'aBaCCb', 'aacBBc', 'aabCBc', 'abaCBc', 'abcBaC', 'aBcccAb', 'aBBBcAc', 'aacbbc', 'aabcccb', 'aBaCbc', 'aacBcAb', 'aaaCBBC', 'abbaCCC', 'abcacb', 'abACBC', 'acBaBCC', 'abCABC', 'aaBcBAc', 'aaBcacB', 'acACBBB', 'acBaBc', 'aaCbcb', 'aBcACB', 'acaBBBC', 'aabcBc', 'aBccAb', 'acABBc', 'aaaBCbC', 'aaaCCbb', 'abCCaCb', 'aaBBcc', 'abCaCb', 'acABcB', 'abbCAbc', 'abacbcc', 'aaBcbc', 'aabacbc', 'aaBBBcc', 'abbaCbC', 'abbcbac', 'aaBBaCC', 'abaccB', 'abCbaC', 'abCCAcb', 'aaCCBBC', 'acbbACb', 'acBcAb', 'abAcccB', 'aabCBC', 'abbCbaC', 'aabbccb', 'aaaCbbC', 'aaabCbc', 'aabAcBc', 'abbCBac', 'aaBBCC', 'acbAbC', 'aacBCb', 'aaaCBcb', 'aBcbaC', 'aaCbbCC', 'abCaCBB', 'aabcacb', 'abcacBB', 'aaaBBcc', 'abccABc', 'acaBCbb', 'aabCCCb', 'aaBBacc', 'aaabbCC', 'acAcbb', 'acbbbaC', 'aBcAcbb', 'accbAB', 'aabbCC', 'acaBCb', 'abcccAB', 'abAccB', 'abaCbC', 'abcbAC', 'acbABc', 'abcacB', 'aBCCaCB', 'acaCbbb', 'aaBCBc', 'accBaBc', 'aaBcccB', 'abaCCB', 'abcABcc', 'aabCbAc', 'abbCBaC', 'aaBcabC', 'acAbcb', 'aaBCbc', 'accbaB'}
500
"""


# Assumption: each word is a power of order at least four of a certain word.
# Only condition (d) of (4.23) (Cyclic polytope) is possible.


