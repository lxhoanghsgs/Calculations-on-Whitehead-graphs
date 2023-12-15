import gurobipy as gp
import math
from gurobipy import GRB
import numpy as np
import itertools
# import igraph
import networkx
# We work over the free group of rank 3.
generator = ["a", "b", "c", "A", "B", "C"]
n = len(generator)
sample_list_of_words = ["abcABcc",]
def get_length_two_subwords(list_of_words):
    list_of_length_two_subwords = []
    for i in list_of_words:
        for j in range(-1, len(i)-1):
            list_of_length_two_subwords.append(i[j] + i[j+1])
    return list_of_length_two_subwords
# print(get_length_two_subwords(sample_list_of_words))
gen_dict = {"a": 0, "A": 1, "b": 2, "B": 3, "c": 4, "C": 5}
reversed_gen_dict = {0: 'a', 1: 'A', 2: 'b', 3: 'B', 4: 'c', 5: 'C'}
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
    for i in range(len(l) - 1):
        if l[i][1].islower():
            dict_of_edge_pairing[l[i][1]].append((i+1 + w_index, i + w_index))
        else:
            m = l[i][1].lower()
            dict_of_edge_pairing[m].append((i + w_index, i+1 + w_index))
    if l[-1][1].islower():
        dict_of_edge_pairing[l[-1][1]].append((w_index, len(l) - 1 + w_index))
    else:
        m = l[-1][1].lower()
        dict_of_edge_pairing[m].append((len(l) - 1 + w_index, w_index))
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
def get_all_cycles(incidence_matrix: np.ndarray) -> dict:
    all_c = dict()
    for v in itertools.product({0, 1}, repeat=len(incidence_matrix[0])):
        resulting_check = np.matmul(incidence_matrix, v)
        if all([i in {0, 2} for i in resulting_check]) and any(i > 0 for i in resulting_check):
            if any(is_subset(cyc1, v) for cyc1 in all_c):
                pass
            else:
                all_c[v] = "".join(reversed_gen_dict[j] for j in range(len(v)) if v[j] == 1)
    return all_c
# A_G, edge_pairing = get_whitehead_graph_with_edge_pairing(sample_list_of_words)
# print(is_minimal_and_diskbusting(sample_list_of_words))
# cycles = get_all_cycles(A_G)
# cycle_mat = np.array(cycles)
# print(cycle_mat)
# print(edge_pairing)

# # gurobi model
# model = gp.Model("Find balanced list of cycles")
# vars = model.addMVar(shape = (1, len(cycles)), vtype = GRB.INTEGER, lb = 0)
# model.addConstr(gp.quicksum(vars) >= 1)
# # conditions for balance
# for gen in ["a", "b", "c"]:
#     for (i, j) in itertools.combinations(edge_pairing[gen], 2):
#         model.addConstr(gp.quicksum(vars[0, k] for k in range(len(cycles)) if cycles[k][i[0]] == 1 and cycles[k][j[0]] == 1) == gp.quicksum(vars[0, k] for k in range(len(cycles)) if cycles[k][i[1]] == 1 and cycles[k][j[1]] == 1))
# # calculations
# model.setObjective(0, sense = GRB.MINIMIZE)
# model.Params.MIPFocus = 1
# # focus on finding feasible solutions
# model.optimize()

# sol = [int(i) for i in model.X]

# Calculatable now.

def get_set_of_all_good_words_of_length_at_most(length : int) -> set:
    # minimal_diskbusting_cyclically_reduced
    res = set()
    for i in range(1, length + 1):
        for j in itertools.product(generator, repeat = i):
            word = "".join(j)
            if is_cyclically_reduced(word) and is_minimal_and_diskbusting([word,]) and res.isdisjoint(get_redundant_words(word)):
                res.add(word)
    return res
def get_set_of_all_good_words_of_length(length : int) -> set:
    # minimal_diskbusting_cyclically_reduced
    res = set()
    i = length
    for j in itertools.product(generator, repeat = i):
        word = "".join(j)
        if is_cyclically_reduced(word) and is_minimal_and_diskbusting([word,]) and res.isdisjoint(get_redundant_words(word)):
            res.add(word)
    return res
"""
checklist2 = get_set_of_all_good_words_of_length(8)
print(checklist2)
{'aBcAcbbb', 'abaBaCAc', 'abbbcAcB', 'abaBcbcB', 'aaaaCBcb', 'abCACBBB', 'acAbcBCb', 'aacBcaBB', 'abcACCCb', 'aaaBAcbc', 'acbbbbaC', 'abCCaCCb', 'aaaBCaCB', 'abaCACaB', 'abCCCCAb', 'aBaCCBcc', 'abcBcbaB', 'aaaBBCCC', 'acbCBaCb', 'aaaabcbC', 'abbbbcaC', 'acbbACbb', 'acABcbAC', 'aaBCCBBB', 'abbCCbaC', 'acaBBBcB', 'acbCaCBc', 'aaCbbCab', 'aaBaCCBC', 'abaBaCAC', 'abcaCbAb', 'abCaBcbc', 'abCAbAcb', 'acbaCbcB', 'acACBCbC', 'abaCCCbc', 'abaccBCC', 'abacBCaB', 'aaCbbabC', 'abCBacaC', 'acbaCbCB', 'aaaCCBBC', 'abaCbCCC', 'abCbbbaC', 'abcBcbAc', 'abbCAbbc', 'abcBCbAC', 'abCbcaBC', 'aaBCBaCC', 'acbaCbAC', 'aaccbbcc', 'aaBCaCCB', 'acAcbCBc', 'aBaCCCbc', 'aBBcbbaC', 'acaBBcbb', 'aBcccaCB', 'abABCbcB', 'acBBcaBB', 'acBCBaCb', 'abCCCaCb', 'abCCaccb', 'acccBaBC', 'aaaaBcBC', 'abCCCCaB', 'aBBCaCCB', 'abbbbCac', 'aaBBaCCC', 'acbaCbAc', 'aaCCbbbb', 'abcBaCAc', 'acBaBccc', 'aBCBBaCC', 'aacBBcaB', 'abACbaCB', 'aaBCCBaC', 'abaCCCBc', 'acACbABC', 'aabCbaCC', 'abcaCbaB', 'abcACbaB', 'aaaCbCAB', 'aabbCbaC', 'abAbCABC', 'abCBBBaC', 'aaaccBBc', 'abbCACBB', 'abcBcbAb', 'aBaCBCCC', 'aaBCaaCB', 'abACBcAB', 'aabcacbb', 'acBCbcAB', 'aaCCCCBB', 'acBcccaB', 'abCaCCbb', 'abcccACb', 'acABcBCB', 'abAbaCBc', 'abCbaBCB', 'abCBaBcb', 'aaaacBCb', 'abCacbaB', 'abacbAbC', 'abaCCbbC', 'aabcccbb', 'acbbbABc', 'abCAcACB', 'aBCCCaCB', 'aaaaCBBC', 'aaaacBCB', 'aaaCABCB', 'abABcccc', 'aabCCbbb', 'acbcAcBc', 'acbCbcAC', 'abCAcbAb', 'aaaabCBC', 'abCAcaCB', 'abAcBCAB', 'abcAcBac', 'aaaCAbcb', 'abacBCCC', 'abbbcbac', 'abbbbCAC', 'abAcbCAb', 'abABaCbc', 'abbbCbaC', 'abABaCAC', 'aaaBcBAc', 'aaaCBBCC', 'abcacbbb', 'acbbbbAC', 'aabbcbac', 'abacaBaC', 'abAbCBcb', 'acbCaCbc', 'aBBBBcAC', 'aaaBccBB', 'abbbbacc', 'aBBBCBaC', 'abAccccb', 'aaBaccBc', 'aaBBBccB', 'abccccaB', 'aaaCbacB', 'aaccbbbc', 'abABaCAc', 'aBBBCbaC', 'aaBBBBCC', 'aaBBBccc', 'aaCCabbb', 'abcbACbC', 'acBCBcAB', 'aaaCbCab', 'aabAAcbc', 'aaccBBBc', 'accaBcBB', 'aaaabCbc', 'abCABaBC', 'abACCCCB', 'accaBBcB', 'aacAABCB', 'aaBCaCBB', 'aaBCCaCB', 'aaabCaCb', 'abcbCbAc', 'abCaCbbb', 'abCAbbbc', 'abcaCbAB', 'aacccabb', 'abcaccbb', 'abaBcaCB', 'abcbbacc', 'abAcBCAb', 'abABacbC', 'aabCCaCb', 'abccbbac', 'accBccaB', 'aabaaCbC', 'abbCCaCb', 'aBCbbbaC', 'aBCCCBaC', 'abCBCaBc', 'aacccaBB', 'abCCCAcb', 'aaaBcacB', 'aaabbCCb', 'aBaCCCBC', 'abAcbCBc', 'aBBBCaCb', 'abcBcaBC', 'aBBCaCbb', 'abcAbaBc', 'aaabCCbb', 'abaBcAbc', 'aaBBaCBC', 'abCBCbaB', 'acAcbACb', 'aaBcBBac', 'abbbaccc', 'abCaBacb', 'abbcaccb', 'abcACacB', 'aacbbbcc', 'aaaBBCCB', 'abbCaCbb', 'acBCBcAc', 'acBcbcAc', 'abbbacbc', 'abcABcaB', 'aaaBACbC', 'abCCbbaC', 'aaCCCabb', 'abcACbAB', 'aaBccBac', 'aaaaCCBB', 'aaaaCbcb', 'aaacbcAB', 'aabacbcc', 'acbaBBBc', 'abcccABc', 'acABCBcB', 'aaaBcBAC', 'abCbaCCC', 'aaccaBBB', 'aaaaCBcB', 'acbbbbAc', 'acaBCbaC', 'abcbCbAb', 'aaacBcAb', 'aaBcaccB', 'aaaCBBBC', 'aaaacBcb', 'abcBBBaC', 'aBaCCCCb', 'aabCbAAc', 'aaBacBBc', 'abcaCacB', 'aBccccAB', 'acbCBcAc', 'abAbaCAC', 'aaCCBBCC', 'abacACaB', 'acccBcaB', 'abccbacc', 'aabbaaCC', 'abACAbaC', 'aaCCbbbC', 'abbacccc', 'abACBcBC', 'abcACAcB', 'abcBacAC', 'abccaccb', 'aaacAbCb', 'abcacccb', 'aaCBBCCC', 'abCbbaCC', 'abcbCbAC', 'abCAbcAc', 'aabAAcBc', 'aaaBBBCC', 'abCacccb', 'acbCBcAC', 'acbcBaCb', 'aBcccACB', 'aaCbbCCC', 'aaaabbcc', 'acaBccBB', 'acaCbaBC', 'aaaCBacb', 'abACaCAB', 'aacBBBcc', 'abCAbABC', 'aaaaBCBc', 'abaCCbcc', 'aaaCCabb', 'abaBaCbc', 'acACaBcb', 'aBcaCCCB', 'aaCbbbCC', 'abCCCacb', 'aaaCbbCC', 'aaaCBCAb', 'aaBCCCBB', 'abaBCAbC', 'abcaCCCb', 'aBBBaCBC', 'aaaaBccB', 'abbaccbc', 'aaabbacc', 'aaacbabc', 'abaBCbCB', 'abCbcbAC', 'acaCBcbC', 'acbbaBBc', 'abaCbbCC', 'abAbacAC', 'acBcbcAb', 'acbAcaCb', 'aabbCCbb', 'abAcaCAb', 'acbCBcAB', 'accccbAb', 'accBaBBc', 'aaaCABcB', 'acAbcBcb', 'aBBaCBCC', 'aaaBACBC', 'aaacBBcc', 'aBaCBBBC', 'aaabbbCC', 'abABcBCB', 'acbcBcAB', 'abbbbcAC', 'abacccBC', 'abcACAbC', 'abacbbbc', 'acbbbACb', 'abAcBcbc', 'abABcAbc', 'accccaBB', 'acBCbcAc', 'aacccbbb', 'aBBBBaCC', 'acAcaBcb', 'acbcABcB', 'abcccaCb', 'abaBacbC', 'aaabACbC', 'aaabcaBC', 'abccccAb', 'abcacBaC', 'abCCCCAB', 'aaaCCaBB', 'aabcbbac', 'aaaacBBc', 'aabcbAAC', 'aaacbcAb', 'abCacbAB', 'acbACaCb', 'aaaaCBCb', 'aaCBaBCC', 'acbCbcAB', 'abaCCCCB', 'abaBacAc', 'aaaacbcB', 'aaabbCCC', 'aaaCbCAb', 'aBCCBBaC', 'aaaCaBCB', 'aabCCbaC', 'abABaCac', 'aBCaCCBB', 'aaabAcbc', 'aaaccaBB', 'abaBacaC', 'abbcBBaC', 'abcBCbAB', 'acaCBcBC', 'accBaBcc', 'abACBaCb', 'aBBBBcAc', 'aabaCbbC', 'abAbaCac', 'abcbCbAB', 'aaacBaBc', 'abAbCBCb', 'abCAbcAC', 'aaabCbAc', 'acbABcAC', 'abACBcbC', 'aabcaaBC', 'abaCAcaB', 'abACABaC', 'aaacBaCb', 'abaBaCac', 'aaBBBCCC', 'aBccAbcc', 'aaCCCaBB', 'acABBBBc', 'acaCbAcb', 'acccbAcB', 'acAbbbbc', 'abbaCbbC', 'abaCBcaB', 'aaaCCbbC', 'abaCBccc', 'abaCbACB', 'abcbCbaB', 'abacccbc', 'abcAbACb', 'acbACbaC', 'abAcBCBc', 'abbccbac', 'aaaBAcBc', 'acAcbCbc', 'abaCbccc', 'abCaCACB', 'abacbCaB', 'aBCCaCCB', 'abACacAb', 'abaBCacB', 'aaaCCbbb', 'abbcAcBB', 'acaCaBcb', 'acbcAbCb', 'abAcacAB', 'abABcbCB', 'abACaCAb', 'acBBBaBc', 'abCbcbAc', 'abCCCABC', 'acbABBBc', 'acACaBCb', 'abcaBcAb', 'abAbCbcb', 'aaBBBacc', 'aabbbaCC', 'aabbCCCb', 'aacaBccB', 'aaaabcBc', 'aaBCBaaC', 'abcbbbac', 'aaaccabb', 'abAcABac', 'acBcbcaC', 'acaCBBBB', 'accBBcaB', 'aaabcbAC', 'abCACaCB', 'acBaCBCb', 'abaCbcaB', 'abcaBcBC', 'aBCCaCBB', 'abbcbacc', 'acAcbbbb', 'abaBcbCB', 'acbCBcaC', 'acAbCBCb', 'aaCCCCbb', 'aaCCaBCB', 'abaBCbcB', 'abaCbAbc', 'abCBaCAc', 'aaabcbac', 'abCAbaBC', 'abbbbcAc', 'aaBBBCCB', 'aaaBCCCB', 'acaCbABC', 'acbCbcAc', 'aBaCBBCC', 'abcBCaBc', 'aaCBBBBC', 'acccaBcB', 'aaaccbbb', 'aaacaBcB', 'aaacccbb', 'aaBBcccB', 'abAcbcBc', 'aaaBBaCC', 'abABcbcB', 'acBCaCbc', 'aabbcccc', 'abCbAbcb', 'abccABcc', 'abCAcbaB', 'abACacAB', 'abCBcbAb', 'abABCAcB', 'acACbbbb', 'abCCCbaC', 'acBcbcAC', 'aBBCBBaC', 'aaabcbAc', 'aaBBcccc', 'aabbaCbC', 'aaccBBBB', 'abAcbCbc', 'abABcaCB', 'abaCbbbC', 'abCaCAbc', 'aBCaCBBB', 'aaBBCCCB', 'abAbacbC', 'abCBcbaB', 'acbABcaC', 'aaBcBacc', 'acaBBBcb', 'aaaccBBB', 'accbAccB', 'aaaaCCbb', 'aaBBCCCC', 'abCaBCBc', 'acACbCBC', 'abACBcAb', 'acbaCBcB', 'abbbbaCC', 'abbCBBac', 'aaaaBcbc', 'abCAcccb', 'aaCCBCaB', 'acbaCaBc', 'abbbcacb', 'acbCbcAb', 'aabbcccb', 'abacAbAc', 'aBBaCBBC', 'abaBacAC', 'aaaCCCBB', 'abcBaBcb', 'aaaCCCbb', 'acccBaBc', 'aaaBBccB', 'acaBcccB', 'acbcBcAb', 'acaCaBCb', 'aabcaccb', 'aaBCBBaC', 'abCBaCac', 'aBBBcAcb', 'aBccccAb', 'abcAcacB', 'aaabbaCC', 'abacAbAC', 'abcaCAcB', 'acaCBCbC', 'abAcbCAB', 'abbbCBaC', 'aabbacbc', 'abbCaCBB', 'aaacbaCB', 'abACbcBC', 'aabcbAAc', 'acbAbCCC', 'aabbcacb', 'aaCBBBCC', 'abcbAcBc', 'acABCbAC', 'aBBBcABC', 'aaaCBaBC', 'acABcaBC', 'abABcACB', 'aacAABcB', 'abCbcbAb', 'aaccccbb', 'abABCacB', 'abbbCACB', 'acaBcAcb', 'aBBBaCCC', 'aabaCbCC', 'aacabbcb', 'aacBccaB', 'acaBCbbb', 'aacBcaaB', 'abacccbC', 'aBcABBBC', 'abcaBaCb', 'abCacbAb', 'acBccaBB', 'abbcbbac', 'aBBBcbaC', 'acAcBBBB', 'abCaBCAB', 'acAcbaBc', 'abAbcBCb', 'aabaacbc', 'aaaaBBcc', 'aabaCCbC', 'abcBcbAC', 'abaBCAcB', 'aabCbbaC', 'abcbCaBc', 'aabCbaaC', 'aaabbccc', 'aaaBcccB', 'abaccccB', 'aacccBBc', 'abCABCCC', 'abaCBaBc', 'acaBcbaC', 'acbcBcAC', 'aaCbabbC', 'aBBcAcbb', 'aBcccAbc', 'aaaabcBC', 'aabbaCCC', 'aBBaCCBC', 'aabbbccc', 'aabccbbb', 'acbAcACb', 'aBBCbbaC', 'aaCCBaBC', 'aaaacbbc', 'abAbcaCb', 'acBCbaCB', 'aacBBBBc', 'aaCBBaBC', 'aaabCbAC', 'abbCBBaC', 'aaCCbabC', 'abAbaCbc', 'aaccbcab', 'abCbcbAB', 'acBBaBcc', 'abbaCbCC', 'accaBBBB', 'accBCCaB', 'abaBCCCC', 'abAbcBcb', 'aBBBBcaC', 'abCbAcbc', 'aaaaBCbC', 'aacbaaCB', 'abcAcBBB', 'aaabCaBc', 'aaccBcaB', 'abACAcAB', 'aaccaaBB', 'acbaBcAC', 'aabbaacc', 'aBccaCCB', 'acABBBBC', 'abaCCbCC', 'aaaBBacc', 'abcBCbAb', 'aaaBCBaC', 'aaacAbcb', 'abCaBCAb', 'abcBCbAc', 'acaBBccB', 'aBBBCaCB', 'aaacABCB', 'aBBCaCBB', 'aaCabbCb', 'acbcACbC', 'acBaCBcb', 'abcaBCbC', 'abAcACAb', 'aacBBaBc', 'acaCbCBC', 'aacbabbc', 'aacBaBBc', 'acbCCCaB', 'aaaabCCb', 'aabbCCCC', 'acABCbCB', 'acAbcbCb', 'abCbACBC', 'abAbcbCb', 'abAcAbac', 'acAcBCbc', 'acaBBCbb', 'aaabcacb', 'abcBacaC', 'aaccccBB', 'abAbacAc', 'aaabccbb', 'aabcbaac', 'aacbabcc', 'aaacabcb', 'aaBcccBB', 'aabCCCbb', 'abbcacBB', 'aaaccbbc', 'abcACbAb', 'aaaBaCBC', 'abAbCacb', 'abbCbbaC', 'acaBcBBB', 'abCacACB', 'acAcbcBc', 'aBcACCCB', 'aaabbbcc', 'aaBaCBBC', 'abaCacaB', 'aBCBaCCC', 'aaacccBB', 'aabaccbc', 'abAcacAb', 'abCaCBBB', 'aaabacbc', 'aBaCCBBC', 'aaCbCCab', 'aaaCCBBB', 'abAccccB', 'aaCabCbb', 'abABacBC', 'aBCaCCCB', 'abacbbcc', 'aaCCBBBC', 'aaCaBBCB', 'acaBBBBC', 'abACbcAB', 'abccaCCb', 'abCBBBac', 'abCBcbAB', 'aaccBaBc', 'abCBcbAC', 'aabbbbCC', 'acbcBcaC', 'abbbCAbc', 'abcAbbbC', 'abCCAccb', 'acBcbcAB', 'abbbcacB', 'aBBCBaCC', 'abaccbcc', 'acABCaBc', 'acbCBcAb', 'aaaaBBCC', 'abACAcAb', 'abcBBBac', 'abaCaBac', 'abACbCBC', 'acccbCaB', 'accccbaB', 'aaccaBcB', 'acABcbAc', 'acBCbcAb', 'abacbAcB', 'aaCbCabb', 'abCCABCC', 'aacbbcab', 'aaacBBBc', 'aabAACBC', 'abcacBBB', 'aaCCbCab', 'acaBcABC', 'abbbcBac', 'aaaBBccc', 'abbcacbb', 'abbbCaCB', 'abCacaCB', 'aaacBcAB', 'aaccBBcc', 'acbcaCbC', 'acBcAbcb', 'aacccbbc', 'acbACbbb', 'aaaaCbcB', 'abbbaCbC', 'aacbcAAB', 'abcBaCAC', 'abcABccc', 'acBaBBBc', 'abcABaBc', 'abcbAbCb', 'abcaBcbC', 'acAbbbbC', 'aBaCCBCC', 'aaacABcB', 'aaCBCaBB', 'abCbABCB', 'abCBcaBC', 'acBCBcaC', 'aacabcbb', 'aaCCaBBB', 'abaBcccc', 'abABCbCB', 'accBBaBc', 'acAbCbcb', 'aacbccab', 'abCAcbAB', 'aaabACBC', 'acACbaBC', 'abAcaCAB', 'aBCaCbbb', 'acbaBcaC', 'aabbbCCb', 'aaaabbCC', 'aabbbCCC', 'acbCbaCB', 'aaBBacBc', 'abABacaC', 'abCAcAbc', 'acBcAcbc', 'abAbCAcb', 'aaaaBCCB', 'aaaBCCBB', 'abcaCAbC', 'aabccccb', 'aaccabbb', 'aaCBCCaB', 'abcAbCAC', 'acBBaBBc', 'acBCbcaC', 'abABacAc', 'abCbcbaB', 'aaCCBBBB', 'aaaBCBAc', 'abbaCCCC', 'acbABcAc', 'aaBBcacB', 'aaCCCBBC', 'aaaBBBcc', 'aaaaCbCB', 'aBBcABBC', 'aabbbbcc', 'aaBaaCBC', 'abCACBaC', 'accccbAB', 'acaCbcbC', 'accBaBCC', 'aaCabCCb', 'aaBBcBac', 'aacbbccc', 'acaCbbbb', 'abbCaCCb', 'aabccacb', 'abbbcAbC', 'abAbCCCC', 'aabCaaCb', 'abCCaCbb', 'acBcaCBC', 'abacaCaB', 'aBcbbbaC', 'abcBcbAB', 'abbCbaCC', 'aaacbbcc', 'aBBCCBaC', 'aaaacbCB', 'aaBccccB', 'aacccBBB', 'abAcBacb', 'acBCBcAb', 'abbbCaCb', 'abCBaBCb', 'aabCaCCb', 'acBcaBBB', 'aaccabcb', 'aacBcAAB', 'aaCCabCb', 'aBaCCCBc', 'aaBBCCBB', 'aabbCaCb', 'acaBBBCb', 'aaabcccb', 'abbaCCbC', 'aaCbbbbC', 'aaabAcBc', 'aaBBBBcc', 'abaCAbAc', 'aaBBccBB', 'acBCBcAC', 'abABaCBc', 'acBaCbcb', 'abcBaBCb', 'aaCCbbCC', 'abAcACAB', 'aacBBccc', 'acACBcbC', 'abCABCaB', 'acBCCCaB', 'aacAAbCb', 'abaBcBCB', 'aabacbbc', 'aaBBCBaC', 'aaCCCbbC', 'acaBCaCb', 'aaCbabCC', 'acABCbcB', 'aBaCBccc', 'aaabbccb', 'aaCaBCCB', 'aaaaccbb', 'aBccACCB', 'aacbbabc', 'aaaacbCb', 'accbAbCC', 'acACbcBC', 'abbacbcc', 'aaBBaccc', 'abcccbac', 'aBBCCaCB', 'abaBacBC', 'acBBBcaB', 'acaBcbbb', 'accbCCaB', 'aaaBacBc', 'aaBccacB', 'aaaCabCb', 'abaBCBcB', 'abbbcBaC', 'aaaaBcbC', 'aaBBCaCB', 'aabbaccc', 'abcbaBcB', 'abcAbCAc', 'acACBBBB', 'acBBBBaC', 'aBCBBBaC', 'abbbaCCC', 'acBaBCCC', 'aabCaaBc', 'acACBcBC', 'acbbABBc', 'acABcbcB', 'aaaaCbbC', 'abCaBCbc', 'aaacbbbc', 'abcccacb', 'abcAbCac', 'abABCAbC', 'abCBCbAC', 'abaCCCbC', 'abABCBcB', 'abaBcACB', 'aaCaBCBB', 'abcacAbC', 'abcAbABc', 'abccacbb', 'aaaCbbbC', 'abaccbCC', 'aaabCCCb', 'aacBaBcc', 'aaacBcaB', 'acbcBcAc', 'acAcaBCb', 'acACbcbC', 'aaBccBBB', 'aaCBaBBC', 'abAbcACb', 'aabbccbb', 'abCBacAc', 'acABCbAc', 'acACbAcb', 'abAcbacB', 'aaaabCBc', 'acAcbABc', 'aacbbbbc', 'abCBCbAc', 'abaCAbAC', 'acbbbaBc', 'acBCaCBc', 'abCAbcaC', 'abcBaCac', 'aBcAbccc', 'abcbABcB', 'abCaCCCb', 'abacBaBC', 'aacbcabb', 'aBBaCCCC', 'aaaBCabc', 'abCBaCAC', 'acBCbcAC', 'abbbbCAc', 'abAbaCAc', 'abcbaccc', 'aaBBaaCC', 'aaaaccBB', 'aabCCCCb', 'aacabccb', 'acaBBcBB', 'acaCbcBC', 'abCBCbAB', 'abaCCBcc', 'abacbccc', 'aaaabccb', 'accaBccB', 'abABacAC', 'acBcACBC', 'abccACCb', 'acAcBcbc', 'aaaCAbCb', 'aabCbAAC', 'aaaCBCaB', 'aabcaacb', 'aacaBBcB', 'aacbcAAb', 'aaBacBcc', 'aacBaaBc', 'abacAcaB', 'acBaBBcc', 'aaBaCBCC', 'abbcAbbC', 'aaccbabc', 'abcaBcAB', 'aacBcAAb', 'aacaaBcB', 'aabCaCbb', 'abAbacBC', 'abABCCCC', 'aaBCCCCB', 'aabbbacc', 'abaBaCBc', 'abACbcAb', 'aaCCCbbb', 'acccBCaB', 'acBBccaB', 'acbAcccB', 'aBCCBaCC', 'aaabaCbC', 'aaCBBCaB', 'aaaCBCAB', 'acccbAbC', 'abCBcbAc', 'acBcABCB', 'aabccbac', 'abcBCbaB', 'aacaBcBB', 'abccccAB', 'aBaCbccc', 'acccaBBB', 'abcBacAc', 'aaaBCBAC', 'aaaBcabC', 'aaaBcBac', 'abcacAcB', 'abAbacaC', 'abAbcABc', 'abCCbaCC', 'acbaBcAc', 'accBcaBB', 'aaBcacBB', 'aabAACbC', 'aabcbacc', 'abCBacAC', 'aaBBBaCC', 'aBaCCbcc', 'acABcbCB', 'abbbCBac', 'abACCCCb', 'abaccbbc', 'abACBCbC', 'abbccacb', 'aaabCbaC', 'abAcBCbc', 'aaaCbabC', 'abbacbbc', 'abCacAbc', 'acbAcABc', 'aacAAbcb', 'aabbbccb', 'aaaaBCbc', 'acAbCBcb', 'acAcBCBc', 'abacbCCC', 'acBcbaCB', 'aaacbcab', 'aaccbbbb', 'abbcBBac', 'abCaCBac', 'abAbcccc', 'abCBCbAb', 'aaCCCBBB', 'abACbcbC', 'aacBaaCb', 'acbCbcaC'}
"""
death = []
def get_gcd_of_list(l: list) -> int:
    res = l[0]
    for i in l:
        res = math.gcd(res, i)
    return res
# checklist1 = {'abbcAcB', 'abCacb', 'aabACbC', 'acABBBC', 'aBaCCbc', 'aBBCBaC', 'acBBcaB', 'aBCaCBB', 'aaacBBc', 'aabcbac', 'abaCbcc', 'aBBcABC', 'abcacbb', 'abbcaC', 'acACbbb', 'acBcaBB', 'abACbC', 'abbCaCB', 'acABBBc', 'abbcAC', 'aaCCaBB', 'abaCCbc', 'aaaBBCC', 'abacbbc', 'aBBcaC', 'aaacbCB', 'aaccaBB', 'acaBBC', 'aBBaCBC', 'aBaCBc', 'abbaccc', 'acaCbb', 'abcBBaC', 'aabCaBc', 'aabbaCC', 'aaBBCCB', 'abCCCaB', 'abaBcc', 'aabcBC', 'acaBBcb', 'aBaCBCC', 'aabbCCC', 'aaCCabb', 'acbAbCC', 'aaaCCBB', 'aaacbbc', 'abCAccb', 'aacbcAB', 'acbaCB', 'aacABCB', 'aabACBC', 'aaBBCCC', 'abCBaC', 'acACbb', 'acAcBBB', 'aaBCCBB', 'aBccACB', 'abacBC', 'abbbCac', 'abCBac', 'abCbbaC', 'aacbcab', 'aacbbbc', 'abACCCB', 'aBCaCbb', 'acAbbc', 'abCCbaC', 'acbbbAc', 'aBcbbaC', 'abccaCb', 'acbACbb', 'abccbac', 'acaBccB', 'abcbAc', 'aabcbAc', 'abAbcc', 'abABCCC', 'abbcBac', 'abaccbc', 'aBBCbaC', 'aaabccb', 'aBCbbaC', 'abcAbC', 'aaCCCBB', 'aaBCCB', 'aaaBCbc', 'acbAcB', 'aaBBccB', 'aBccAB', 'aaacBCB', 'abbbacc', 'aaccbbb', 'aaCCbbb', 'acABBC', 'aabCbAC', 'aaaCbCB', 'acAcBB', 'aaccBB', 'abAcbc', 'aacccbb', 'aaccBBB', 'abABcc', 'aaBCCCB', 'abbcacb', 'aBCCBaC', 'abaccBC', 'aacbCb', 'abaCbc', 'aacAbcb', 'aBaCCBc', 'abCBBac', 'aabcbAC', 'aBBBcAC', 'aaabCCb', 'abbCAc', 'acccbAb', 'aabbacc', 'acBcaB', 'aabaCbC', 'acBBaC', 'abACCb', 'aabAcbc', 'aaBccBB', 'aabccb', 'aaCCbbC', 'abABCC', 'aacbaCB', 'aacBcaB', 'acccbaB', 'abbCac', 'aaabcBC', 'aaBaCBC', 'acaBcb', 'aBBcAcb', 'aacbcB', 'aaaCbcB', 'aBBcbaC', 'aBcAcb', 'aabCCbb', 'aabbbcc', 'abacbC', 'acAbbbc', 'aaaccbb', 'abcaccb', 'aBcccAB', 'aaBCBAc', 'accBCaB', 'acBaCb', 'abbCAC', 'abccAB', 'aBaCCBC', 'aBaCBC', 'aBaCbcc', 'abaBCCC', 'aacBaBc', 'acBaBcc', 'abaCbbC', 'abbCaCb', 'aabCbaC', 'aaCbCAB', 'accbCaB', 'abbcBaC', 'acAbbC', 'acbbaBc', 'abccaB', 'abaCCbC', 'aBBaCC', 'abAcBc', 'aBcACCB', 'aaaBCBc', 'aBBcAC', 'abCAbc', 'aaCBacb', 'aaCBBC', 'aaCCBBB', 'aacBCB', 'abaccbC', 'abccACb', 'acAbCb', 'abCABCC', 'aaCAbcb', 'abcaCCb', 'abCCABC', 'acBccaB', 'aacbabc', 'aacBBcc', 'abcACCb', 'aabbcc', 'aBCaCb', 'abaBccc', 'aaCabCb', 'abCaCB', 'aBBcAc', 'aaabcbC', 'aaBCabc', 'aaCbCab', 'abcABc', 'aBCBaCC', 'abAccb', 'acBcAB', 'aabbbCC', 'aaCABCB', 'acbbAC', 'accaBB', 'aacaBcB', 'aaCbbbC', 'abcBBac', 'abCCCAb', 'aBcaCB', 'aacABcB', 'aaCaBCB', 'aacBaCb', 'aaCbcB', 'aaCBCb', 'abbbaCC', 'aaabCBc', 'aacbCB', 'acBCCaB', 'aaBacBc', 'aaaCbcb', 'abaCbCC', 'abaCCBc', 'aBccaCB', 'aabcaBC', 'aaBCBAC', 'aBccAbc', 'abcccaB', 'aBcAbc', 'aBaCCCb', 'abCaBc', 'abCbAc', 'abcAcBB', 'abCCAB', 'aaBcBAC', 'abaBCC', 'acaBBCb', 'acaBBcB', 'aaaCBcB', 'aaabbcc', 'aabCCb', 'aaaBcBC', 'abCACBB', 'aBCaCCB', 'aabCbc', 'acABCB', 'abAbccc', 'abcbac', 'acBCaB', 'aaaBcbC', 'abCCAb', 'aaaCBCb', 'abbacbc', 'aBBCaCB', 'aaCBcB', 'acbCCaB', 'acAcbbb', 'aabccbb', 'aaCbCAb', 'aabbccc', 'aacabcb', 'aaCBCAb', 'acbABBc', 'aBaCBcc', 'abCACB', 'abACCB', 'aaBcBC', 'aaaccBB', 'aaCABcB', 'abCbaCC', 'acBBaBc', 'acbCaB', 'abccAb', 'aaCBBCC', 'acbbaC', 'aabCaCb', 'abCCCAB', 'acbbABc', 'aaCBCAB', 'aaccabb', 'aaCBBBC', 'acbaBc', 'accBcaB', 'aBcABC', 'aaacbcB', 'acccbAB', 'aaBACbC', 'aBCBBaC', 'abCAcb', 'acbbAc', 'aaBcbC', 'abbcAc', 'abacbc', 'abacbCC', 'abbaCC', 'acbaBBc', 'abcAcB', 'abbbcAc', 'abaCCCB', 'abCBBaC', 'aaCbacB', 'abbcAbC', 'abcAbbC', 'abAbCCC', 'abaCBcc', 'abbbcAC', 'aaCBcb', 'abABccc', 'aaacBCb', 'abbbcaC', 'aaCCBB', 'aaBBBCC', 'abcbbac', 'abCaCbb', 'abCCacb', 'aaacbCb', 'acaBcBB', 'aaCBaBC', 'abbCACB', 'aBBaCCC', 'abAcccb', 'abbacc', 'accBaBC', 'aaBAcBc', 'aBCBaC', 'abcccAb', 'aaCAbCb', 'aaabcBc', 'aBBBcaC', 'accaBBB', 'accaBcB', 'acaCBBB', 'abbbCAC', 'aabcbC', 'aBcAbcc', 'aBBBaCC', 'abacccB', 'acACBB', 'aaBCbC', 'acccaBB', 'acBaBC', 'aacBcAB', 'aaBAcbc', 'aaCCbb', 'abcbacc', 'aaccBBc', 'aacAbCb', 'abcACb', 'aacbbcc', 'acbbbAC', 'aacccBB', 'abbcacB', 'abAbCC', 'abbbCAc', 'abacBCC', 'accbAbC', 'accbAcB', 'aaccbbc', 'aBCaCB', 'abCAbbc', 'acaBcbb', 'aBCbaC', 'aaaBccB', 'aaCbCB', 'acbAccB', 'abcaCb', 'accbAb', 'abCaCCb', 'acbACb', 'aaCbbC', 'acaBcB', 'aaBccB', 'aabbCCb', 'acbcAB', 'abCaccb', 'aaacBcb', 'acbcAb', 'aacBcb', 'aaabCBC', 'aBcABBC', 'abCCaB', 'aaCCCbb', 'aaCBCaB', 'aBcaCCB', 'aaaBcbc', 'aaBcBac', 'acBBBaC', 'aaaBCCB', 'abACCCb', 'aBBCaCb', 'abCbAC', 'aaBCBaC', 'abcaBC', 'aaBCaCB', 'abccacb', 'aaccbb', 'abcBac', 'acAbbbC', 'aBaCBBC', 'acaCBB', 'aaBACBC', 'aacBBBc', 'aaBBccc', 'acBaBBc', 'aaCbabC', 'aacbcAb', 'aBaCCb', 'aacBBc', 'aabCBc', 'abaCBc', 'abcBaC', 'aBcccAb', 'aBBBcAc', 'aacbbc', 'aabcccb', 'aBaCbc', 'aacBcAb', 'aaaCBBC', 'abbaCCC', 'abcacb', 'abACBC', 'acBaBCC', 'abCABC', 'aaBcBAc', 'aaBcacB', 'acACBBB', 'acBaBc', 'aaCbcb', 'aBcACB', 'acaBBBC', 'aabcBc', 'aBccAb', 'acABBc', 'aaaBCbC', 'aaaCCbb', 'abCCaCb', 'aaBBcc', 'abCaCb', 'acABcB', 'abbCAbc', 'abacbcc', 'aaBcbc', 'aabacbc', 'aaBBBcc', 'abbaCbC', 'abbcbac', 'aaBBaCC', 'abaccB', 'abCbaC', 'abCCAcb', 'aaCCBBC', 'acbbACb', 'acBcAb', 'abAcccB', 'aabCBC', 'abbCbaC', 'aabbccb', 'aaaCbbC', 'aaabCbc', 'aabAcBc', 'abbCBac', 'aaBBCC', 'acbAbC', 'aacBCb', 'aaaCBcb', 'aBcbaC', 'aaCbbCC', 'abCaCBB', 'aabcacb', 'abcacBB', 'aaaBBcc', 'abccABc', 'acaBCbb', 'aabCCCb', 'aaBBacc', 'aaabbCC', 'acAcbb', 'acbbbaC', 'aBcAcbb', 'accbAB', 'aabbCC', 'acaBCb', 'abcccAB', 'abAccB', 'abaCbC', 'abcbAC', 'acbABc', 'abcacB', 'aBCCaCB', 'acaCbbb', 'aaBCBc', 'accBaBc', 'aaBcccB', 'abaCCB', 'abcABcc', 'aabCbAc', 'abbCBaC', 'aaBcabC', 'acAbcb', 'aaBCbc', 'accbaB'}
# checklist2 = {'aBcAcbbb', 'abaBaCAc', 'abbbcAcB', 'abaBcbcB', 'aaaaCBcb', 'abCACBBB', 'acAbcBCb', 'aacBcaBB', 'abcACCCb', 'aaaBAcbc', 'acbbbbaC', 'abCCaCCb', 'aaaBCaCB', 'abaCACaB', 'abCCCCAb', 'aBaCCBcc', 'abcBcbaB', 'aaaBBCCC', 'acbCBaCb', 'aaaabcbC', 'abbbbcaC', 'acbbACbb', 'acABcbAC', 'aaBCCBBB', 'abbCCbaC', 'acaBBBcB', 'acbCaCBc', 'aaCbbCab', 'aaBaCCBC', 'abaBaCAC', 'abcaCbAb', 'abCaBcbc', 'abCAbAcb', 'acbaCbcB', 'acACBCbC', 'abaCCCbc', 'abaccBCC', 'abacBCaB', 'aaCbbabC', 'abCBacaC', 'acbaCbCB', 'aaaCCBBC', 'abaCbCCC', 'abCbbbaC', 'abcBcbAc', 'abbCAbbc', 'abcBCbAC', 'abCbcaBC', 'aaBCBaCC', 'acbaCbAC', 'aaccbbcc', 'aaBCaCCB', 'acAcbCBc', 'aBaCCCbc', 'aBBcbbaC', 'acaBBcbb', 'aBcccaCB', 'abABCbcB', 'acBBcaBB', 'acBCBaCb', 'abCCCaCb', 'abCCaccb', 'acccBaBC', 'aaaaBcBC', 'abCCCCaB', 'aBBCaCCB', 'abbbbCac', 'aaBBaCCC', 'acbaCbAc', 'aaCCbbbb', 'abcBaCAc', 'acBaBccc', 'aBCBBaCC', 'aacBBcaB', 'abACbaCB', 'aaBCCBaC', 'abaCCCBc', 'acACbABC', 'aabCbaCC', 'abcaCbaB', 'abcACbaB', 'aaaCbCAB', 'aabbCbaC', 'abAbCABC', 'abCBBBaC', 'aaaccBBc', 'abbCACBB', 'abcBcbAb', 'aBaCBCCC', 'aaBCaaCB', 'abACBcAB', 'aabcacbb', 'acBCbcAB', 'aaCCCCBB', 'acBcccaB', 'abCaCCbb', 'abcccACb', 'acABcBCB', 'abAbaCBc', 'abCbaBCB', 'abCBaBcb', 'aaaacBCb', 'abCacbaB', 'abacbAbC', 'abaCCbbC', 'aabcccbb', 'acbbbABc', 'abCAcACB', 'aBCCCaCB', 'aaaaCBBC', 'aaaacBCB', 'aaaCABCB', 'abABcccc', 'aabCCbbb', 'acbcAcBc', 'acbCbcAC', 'abCAcbAb', 'aaaabCBC', 'abCAcaCB', 'abAcBCAB', 'abcAcBac', 'aaaCAbcb', 'abacBCCC', 'abbbcbac', 'abbbbCAC', 'abAcbCAb', 'abABaCbc', 'abbbCbaC', 'abABaCAC', 'aaaBcBAc', 'aaaCBBCC', 'abcacbbb', 'acbbbbAC', 'aabbcbac', 'abacaBaC', 'abAbCBcb', 'acbCaCbc', 'aBBBBcAC', 'aaaBccBB', 'abbbbacc', 'aBBBCBaC', 'abAccccb', 'aaBaccBc', 'aaBBBccB', 'abccccaB', 'aaaCbacB', 'aaccbbbc', 'abABaCAc', 'aBBBCbaC', 'aaBBBBCC', 'aaBBBccc', 'aaCCabbb', 'abcbACbC', 'acBCBcAB', 'aaaCbCab', 'aabAAcbc', 'aaccBBBc', 'accaBcBB', 'aaaabCbc', 'abCABaBC', 'abACCCCB', 'accaBBcB', 'aacAABCB', 'aaBCaCBB', 'aaBCCaCB', 'aaabCaCb', 'abcbCbAc', 'abCaCbbb', 'abCAbbbc', 'abcaCbAB', 'aacccabb', 'abcaccbb', 'abaBcaCB', 'abcbbacc', 'abAcBCAb', 'abABacbC', 'aabCCaCb', 'abccbbac', 'accBccaB', 'aabaaCbC', 'abbCCaCb', 'aBCbbbaC', 'aBCCCBaC', 'abCBCaBc', 'aacccaBB', 'abCCCAcb', 'aaaBcacB', 'aaabbCCb', 'aBaCCCBC', 'abAcbCBc', 'aBBBCaCb', 'abcBcaBC', 'aBBCaCbb', 'abcAbaBc', 'aaabCCbb', 'abaBcAbc', 'aaBBaCBC', 'abCBCbaB', 'acAcbACb', 'aaBcBBac', 'abbbaccc', 'abCaBacb', 'abbcaccb', 'abcACacB', 'aacbbbcc', 'aaaBBCCB', 'abbCaCbb', 'acBCBcAc', 'acBcbcAc', 'abbbacbc', 'abcABcaB', 'aaaBACbC', 'abCCbbaC', 'aaCCCabb', 'abcACbAB', 'aaBccBac', 'aaaaCCBB', 'aaaaCbcb', 'aaacbcAB', 'aabacbcc', 'acbaBBBc', 'abcccABc', 'acABCBcB', 'aaaBcBAC', 'abCbaCCC', 'aaccaBBB', 'aaaaCBcB', 'acbbbbAc', 'acaBCbaC', 'abcbCbAb', 'aaacBcAb', 'aaBcaccB', 'aaaCBBBC', 'aaaacBcb', 'abcBBBaC', 'aBaCCCCb', 'aabCbAAc', 'aaBacBBc', 'abcaCacB', 'aBccccAB', 'acbCBcAc', 'abAbaCAC', 'aaCCBBCC', 'abacACaB', 'acccBcaB', 'abccbacc', 'aabbaaCC', 'abACAbaC', 'aaCCbbbC', 'abbacccc', 'abACBcBC', 'abcACAcB', 'abcBacAC', 'abccaccb', 'aaacAbCb', 'abcacccb', 'aaCBBCCC', 'abCbbaCC', 'abcbCbAC', 'abCAbcAc', 'aabAAcBc', 'aaaBBBCC', 'abCacccb', 'acbCBcAC', 'acbcBaCb', 'aBcccACB', 'aaCbbCCC', 'aaaabbcc', 'acaBccBB', 'acaCbaBC', 'aaaCBacb', 'abACaCAB', 'aacBBBcc', 'abCAbABC', 'aaaaBCBc', 'abaCCbcc', 'aaaCCabb', 'abaBaCbc', 'acACaBcb', 'aBcaCCCB', 'aaCbbbCC', 'abCCCacb', 'aaaCbbCC', 'aaaCBCAb', 'aaBCCCBB', 'abaBCAbC', 'abcaCCCb', 'aBBBaCBC', 'aaaaBccB', 'abbaccbc', 'aaabbacc', 'aaacbabc', 'abaBCbCB', 'abCbcbAC', 'acaCBcbC', 'acbbaBBc', 'abaCbbCC', 'abAbacAC', 'acBcbcAb', 'acbAcaCb', 'aabbCCbb', 'abAcaCAb', 'acbCBcAB', 'accccbAb', 'accBaBBc', 'aaaCABcB', 'acAbcBcb', 'aBBaCBCC', 'aaaBACBC', 'aaacBBcc', 'aBaCBBBC', 'aaabbbCC', 'abABcBCB', 'acbcBcAB', 'abbbbcAC', 'abacccBC', 'abcACAbC', 'abacbbbc', 'acbbbACb', 'abAcBcbc', 'abABcAbc', 'accccaBB', 'acBCbcAc', 'aacccbbb', 'aBBBBaCC', 'acAcaBcb', 'acbcABcB', 'abcccaCb', 'abaBacbC', 'aaabACbC', 'aaabcaBC', 'abccccAb', 'abcacBaC', 'abCCCCAB', 'aaaCCaBB', 'aabcbbac', 'aaaacBBc', 'aabcbAAC', 'aaacbcAb', 'abCacbAB', 'acbACaCb', 'aaaaCBCb', 'aaCBaBCC', 'acbCbcAB', 'abaCCCCB', 'abaBacAc', 'aaaacbcB', 'aaabbCCC', 'aaaCbCAb', 'aBCCBBaC', 'aaaCaBCB', 'aabCCbaC', 'abABaCac', 'aBCaCCBB', 'aaabAcbc', 'aaaccaBB', 'abaBacaC', 'abbcBBaC', 'abcBCbAB', 'acaCBcBC', 'accBaBcc', 'abACBaCb', 'aBBBBcAc', 'aabaCbbC', 'abAbaCac', 'abcbCbAB', 'aaacBaBc', 'abAbCBCb', 'abCAbcAC', 'aaabCbAc', 'acbABcAC', 'abACBcbC', 'aabcaaBC', 'abaCAcaB', 'abACABaC', 'aaacBaCb', 'abaBaCac', 'aaBBBCCC', 'aBccAbcc', 'aaCCCaBB', 'acABBBBc', 'acaCbAcb', 'acccbAcB', 'acAbbbbc', 'abbaCbbC', 'abaCBcaB', 'aaaCCbbC', 'abaCBccc', 'abaCbACB', 'abcbCbaB', 'abacccbc', 'abcAbACb', 'acbACbaC', 'abAcBCBc', 'abbccbac', 'aaaBAcBc', 'acAcbCbc', 'abaCbccc', 'abCaCACB', 'abacbCaB', 'aBCCaCCB', 'abACacAb', 'abaBCacB', 'aaaCCbbb', 'abbcAcBB', 'acaCaBcb', 'acbcAbCb', 'abAcacAB', 'abABcbCB', 'abACaCAb', 'acBBBaBc', 'abCbcbAc', 'abCCCABC', 'acbABBBc', 'acACaBCb', 'abcaBcAb', 'abAbCbcb', 'aaBBBacc', 'aabbbaCC', 'aabbCCCb', 'aacaBccB', 'aaaabcBc', 'aaBCBaaC', 'abcbbbac', 'aaaccabb', 'abAcABac', 'acBcbcaC', 'acaCBBBB', 'accBBcaB', 'aaabcbAC', 'abCACaCB', 'acBaCBCb', 'abaCbcaB', 'abcaBcBC', 'aBCCaCBB', 'abbcbacc', 'acAcbbbb', 'abaBcbCB', 'acbCBcaC', 'acAbCBCb', 'aaCCCCbb', 'aaCCaBCB', 'abaBCbcB', 'abaCbAbc', 'abCBaCAc', 'aaabcbac', 'abCAbaBC', 'abbbbcAc', 'aaBBBCCB', 'aaaBCCCB', 'acaCbABC', 'acbCbcAc', 'aBaCBBCC', 'abcBCaBc', 'aaCBBBBC', 'acccaBcB', 'aaaccbbb', 'aaacaBcB', 'aaacccbb', 'aaBBcccB', 'abAcbcBc', 'aaaBBaCC', 'abABcbcB', 'acBCaCbc', 'aabbcccc', 'abCbAbcb', 'abccABcc', 'abCAcbaB', 'abACacAB', 'abCBcbAb', 'abABCAcB', 'acACbbbb', 'abCCCbaC', 'acBcbcAC', 'aBBCBBaC', 'aaabcbAc', 'aaBBcccc', 'aabbaCbC', 'aaccBBBB', 'abAcbCbc', 'abABcaCB', 'abaCbbbC', 'abCaCAbc', 'aBCaCBBB', 'aaBBCCCB', 'abAbacbC', 'abCBcbaB', 'acbABcaC', 'aaBcBacc', 'acaBBBcb', 'aaaccBBB', 'accbAccB', 'aaaaCCbb', 'aaBBCCCC', 'abCaBCBc', 'acACbCBC', 'abACBcAb', 'acbaCBcB', 'abbbbaCC', 'abbCBBac', 'aaaaBcbc', 'abCAcccb', 'aaCCBCaB', 'acbaCaBc', 'abbbcacb', 'acbCbcAb', 'aabbcccb', 'abacAbAc', 'aBBaCBBC', 'abaBacAC', 'aaaCCCBB', 'abcBaBcb', 'aaaCCCbb', 'acccBaBc', 'aaaBBccB', 'acaBcccB', 'acbcBcAb', 'acaCaBCb', 'aabcaccb', 'aaBCBBaC', 'abCBaCac', 'aBBBcAcb', 'aBccccAb', 'abcAcacB', 'aaabbaCC', 'abacAbAC', 'abcaCAcB', 'acaCBCbC', 'abAcbCAB', 'abbbCBaC', 'aabbacbc', 'abbCaCBB', 'aaacbaCB', 'abACbcBC', 'aabcbAAc', 'acbAbCCC', 'aabbcacb', 'aaCBBBCC', 'abcbAcBc', 'acABCbAC', 'aBBBcABC', 'aaaCBaBC', 'acABcaBC', 'abABcACB', 'aacAABcB', 'abCbcbAb', 'aaccccbb', 'abABCacB', 'abbbCACB', 'acaBcAcb', 'aBBBaCCC', 'aabaCbCC', 'aacabbcb', 'aacBccaB', 'acaBCbbb', 'aacBcaaB', 'abacccbC', 'aBcABBBC', 'abcaBaCb', 'abCacbAb', 'acBccaBB', 'abbcbbac', 'aBBBcbaC', 'acAcBBBB', 'abCaBCAB', 'acAcbaBc', 'abAbcBCb', 'aabaacbc', 'aaaaBBcc', 'aabaCCbC', 'abcBcbAC', 'abaBCAcB', 'aabCbbaC', 'abcbCaBc', 'aabCbaaC', 'aaabbccc', 'aaaBcccB', 'abaccccB', 'aacccBBc', 'abCABCCC', 'abaCBaBc', 'acaBcbaC', 'acbcBcAC', 'aaCbabbC', 'aBBcAcbb', 'aBcccAbc', 'aaaabcBC', 'aabbaCCC', 'aBBaCCBC', 'aabbbccc', 'aabccbbb', 'acbAcACb', 'aBBCbbaC', 'aaCCBaBC', 'aaaacbbc', 'abAbcaCb', 'acBCbaCB', 'aacBBBBc', 'aaCBBaBC', 'aaabCbAC', 'abbCBBaC', 'aaCCbabC', 'abAbaCbc', 'aaccbcab', 'abCbcbAB', 'acBBaBcc', 'abbaCbCC', 'accaBBBB', 'accBCCaB', 'abaBCCCC', 'abAbcBcb', 'aBBBBcaC', 'abCbAcbc', 'aaaaBCbC', 'aacbaaCB', 'abcAcBBB', 'aaabCaBc', 'aaccBcaB', 'abACAcAB', 'aaccaaBB', 'acbaBcAC', 'aabbaacc', 'aBccaCCB', 'acABBBBC', 'abaCCbCC', 'aaaBBacc', 'abcBCbAb', 'aaaBCBaC', 'aaacAbcb', 'abCaBCAb', 'abcBCbAc', 'acaBBccB', 'aBBBCaCB', 'aaacABCB', 'aBBCaCBB', 'aaCabbCb', 'acbcACbC', 'acBaCBcb', 'abcaBCbC', 'abAcACAb', 'aacBBaBc', 'acaCbCBC', 'aacbabbc', 'aacBaBBc', 'acbCCCaB', 'aaaabCCb', 'aabbCCCC', 'acABCbCB', 'acAbcbCb', 'abCbACBC', 'abAbcbCb', 'abAcAbac', 'acAcBCbc', 'acaBBCbb', 'aaabcacb', 'abcBacaC', 'aaccccBB', 'abAbacAc', 'aaabccbb', 'aabcbaac', 'aacbabcc', 'aaacabcb', 'aaBcccBB', 'aabCCCbb', 'abbcacBB', 'aaaccbbc', 'abcACbAb', 'aaaBaCBC', 'abAbCacb', 'abbCbbaC', 'acaBcBBB', 'abCacACB', 'acAcbcBc', 'aBcACCCB', 'aaabbbcc', 'aaBaCBBC', 'abaCacaB', 'aBCBaCCC', 'aaacccBB', 'aabaccbc', 'abAcacAb', 'abCaCBBB', 'aaabacbc', 'aBaCCBBC', 'aaCbCCab', 'aaaCCBBB', 'abAccccB', 'aaCabCbb', 'abABacBC', 'aBCaCCCB', 'abacbbcc', 'aaCCBBBC', 'aaCaBBCB', 'acaBBBBC', 'abACbcAB', 'abccaCCb', 'abCBBBac', 'abCBcbAB', 'aaccBaBc', 'abCBcbAC', 'aabbbbCC', 'acbcBcaC', 'abbbCAbc', 'abcAbbbC', 'abCCAccb', 'acBcbcAB', 'abbbcacB', 'aBBCBaCC', 'abaccbcc', 'acABCaBc', 'acbCBcAb', 'aaaaBBCC', 'abACAcAb', 'abcBBBac', 'abaCaBac', 'abACbCBC', 'acccbCaB', 'accccbaB', 'aaccaBcB', 'acABcbAc', 'acBCbcAb', 'abacbAcB', 'aaCbCabb', 'abCCABCC', 'aacbbcab', 'aaacBBBc', 'aabAACBC', 'abcacBBB', 'aaCCbCab', 'acaBcABC', 'abbbcBac', 'aaaBBccc', 'abbcacbb', 'abbbCaCB', 'abCacaCB', 'aaacBcAB', 'aaccBBcc', 'acbcaCbC', 'acBcAbcb', 'aacccbbc', 'acbACbbb', 'aaaaCbcB', 'abbbaCbC', 'aacbcAAB', 'abcBaCAC', 'abcABccc', 'acBaBBBc', 'abcABaBc', 'abcbAbCb', 'abcaBcbC', 'acAbbbbC', 'aBaCCBCC', 'aaacABcB', 'aaCBCaBB', 'abCbABCB', 'abCBcaBC', 'acBCBcaC', 'aacabcbb', 'aaCCaBBB', 'abaBcccc', 'abABCbCB', 'accBBaBc', 'acAbCbcb', 'aacbccab', 'abCAcbAB', 'aaabACBC', 'acACbaBC', 'abAcaCAB', 'aBCaCbbb', 'acbaBcaC', 'aabbbCCb', 'aaaabbCC', 'aabbbCCC', 'acbCbaCB', 'aaBBacBc', 'abABacaC', 'abCAcAbc', 'acBcAcbc', 'abAbCAcb', 'aaaaBCCB', 'aaaBCCBB', 'abcaCAbC', 'aabccccb', 'aaccabbb', 'aaCBCCaB', 'abcAbCAC', 'acBBaBBc', 'acBCbcaC', 'abABacAc', 'abCbcbaB', 'aaCCBBBB', 'aaaBCBAc', 'abbaCCCC', 'acbABcAc', 'aaBBcacB', 'aaCCCBBC', 'aaaBBBcc', 'aaaaCbCB', 'aBBcABBC', 'aabbbbcc', 'aaBaaCBC', 'abCACBaC', 'accccbAB', 'acaCbcbC', 'accBaBCC', 'aaCabCCb', 'aaBBcBac', 'aacbbccc', 'acaCbbbb', 'abbCaCCb', 'aabccacb', 'abbbcAbC', 'abAbCCCC', 'aabCaaCb', 'abCCaCbb', 'acBcaCBC', 'abacaCaB', 'aBcbbbaC', 'abcBcbAB', 'abbCbaCC', 'aaacbbcc', 'aBBCCBaC', 'aaaacbCB', 'aaBccccB', 'aacccBBB', 'abAcBacb', 'acBCBcAb', 'abbbCaCb', 'abCBaBCb', 'aabCaCCb', 'acBcaBBB', 'aaccabcb', 'aacBcAAB', 'aaCCabCb', 'aBaCCCBc', 'aaBBCCBB', 'aabbCaCb', 'acaBBBCb', 'aaabcccb', 'abbaCCbC', 'aaCbbbbC', 'aaabAcBc', 'aaBBBBcc', 'abaCAbAc', 'aaBBccBB', 'acBCBcAC', 'abABaCBc', 'acBaCbcb', 'abcBaBCb', 'aaCCbbCC', 'abAcACAB', 'aacBBccc', 'acACBcbC', 'abCABCaB', 'acBCCCaB', 'aacAAbCb', 'abaBcBCB', 'aabacbbc', 'aaBBCBaC', 'aaCCCbbC', 'acaBCaCb', 'aaCbabCC', 'acABCbcB', 'aBaCBccc', 'aaabbccb', 'aaCaBCCB', 'aaaaccbb', 'aBccACCB', 'aacbbabc', 'aaaacbCb', 'accbAbCC', 'acACbcBC', 'abbacbcc', 'aaBBaccc', 'abcccbac', 'aBBCCaCB', 'abaBacBC', 'acBBBcaB', 'acaBcbbb', 'accbCCaB', 'aaaBacBc', 'aaBccacB', 'aaaCabCb', 'abaBCBcB', 'abbbcBaC', 'aaaaBcbC', 'aaBBCaCB', 'aabbaccc', 'abcbaBcB', 'abcAbCAc', 'acACBBBB', 'acBBBBaC', 'aBCBBBaC', 'abbbaCCC', 'acBaBCCC', 'aabCaaBc', 'acACBcBC', 'acbbABBc', 'acABcbcB', 'aaaaCbbC', 'abCaBCbc', 'aaacbbbc', 'abcccacb', 'abcAbCac', 'abABCAbC', 'abCBCbAC', 'abaCCCbC', 'abABCBcB', 'abaBcACB', 'aaCaBCBB', 'abcacAbC', 'abcAbABc', 'abccacbb', 'aaaCbbbC', 'abaccbCC', 'aaabCCCb', 'aacBaBcc', 'aaacBcaB', 'acbcBcAc', 'acAcaBCb', 'acACbcbC', 'aaBccBBB', 'aaCBaBBC', 'abAbcACb', 'aabbccbb', 'abCBacAc', 'acABCbAc', 'acACbAcb', 'abAcbacB', 'aaaabCBc', 'acAcbABc', 'aacbbbbc', 'abCBCbAc', 'abaCAbAC', 'acbbbaBc', 'acBCaCBc', 'abCAbcaC', 'abcBaCac', 'aBcAbccc', 'abcbABcB', 'abCaCCCb', 'abacBaBC', 'aacbcabb', 'aBBaCCCC', 'aaaBCabc', 'abCBaCAC', 'acBCbcAC', 'abbbbCAc', 'abAbaCAc', 'abcbaccc', 'aaBBaaCC', 'aaaaccBB', 'aabCCCCb', 'aacabccb', 'acaBBcBB', 'acaCbcBC', 'abCBCbAB', 'abaCCBcc', 'abacbccc', 'aaaabccb', 'accaBccB', 'abABacAC', 'acBcACBC', 'abccACCb', 'acAcBcbc', 'aaaCAbCb', 'aabCbAAC', 'aaaCBCaB', 'aabcaacb', 'aacaBBcB', 'aacbcAAb', 'aaBacBcc', 'aacBaaBc', 'abacAcaB', 'acBaBBcc', 'aaBaCBCC', 'abbcAbbC', 'aaccbabc', 'abcaBcAB', 'aacBcAAb', 'aacaaBcB', 'aabCaCbb', 'abAbacBC', 'abABCCCC', 'aaBCCCCB', 'aabbbacc', 'abaBaCBc', 'abACbcAb', 'aaCCCbbb', 'acccBCaB', 'acBBccaB', 'acbAcccB', 'aBCCBaCC', 'aaabaCbC', 'aaCBBCaB', 'aaaCBCAB', 'acccbAbC', 'abCBcbAc', 'acBcABCB', 'aabccbac', 'abcBCbaB', 'aacaBcBB', 'abccccAB', 'aBaCbccc', 'acccaBBB', 'abcBacAc', 'aaaBCBAC', 'aaaBcabC', 'aaaBcBac', 'abcacAcB', 'abAbacaC', 'abAbcABc', 'abCCbaCC', 'acbaBcAc', 'accBcaBB', 'aaBcacBB', 'aabAACbC', 'aabcbacc', 'abCBacAC', 'aaBBBaCC', 'aBaCCbcc', 'acABcbCB', 'abbbCBac', 'abACCCCb', 'abaccbbc', 'abACBCbC', 'abbccacb', 'aaabCbaC', 'abAcBCbc', 'aaaCbabC', 'abbacbbc', 'abCacAbc', 'acbAcABc', 'aacAAbcb', 'aabbbccb', 'aaaaBCbc', 'acAbCBcb', 'acAcBCBc', 'abacbCCC', 'acBcbaCB', 'aaacbcab', 'aaccbbbb', 'abbcBBac', 'abCaCBac', 'abAbcccc', 'abCBCbAb', 'aaCCCBBB', 'abACbcbC', 'aacBaaCb', 'acbCbcaC'}
checklist3 = get_set_of_all_good_words_of_length(7)
sol_dict = dict()
for w in checklist3:
    l = [w,]
    A_G, edge_pairing = get_whitehead_graph_with_edge_pairing(l)
    cycles = get_all_cycles(A_G)
    cycle_mat = np.array(cycles)
    # gurobi model
    with gp.Env(empty = True) as env:
        env.setParam('LogToConsole', 0)
        # not printing logs to console.
        env.start()
        with gp.Model(env = env) as model:
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
            try:
                sol = [int(i) for i in model.X]
                M = get_gcd_of_list(sol)
                sol_dict[w] = [i/M for i in sol]
            except:
                death.append(w)

print(sol_dict)


# Assumption: each word is a power of order at least four of a certain word.
# Only condition (d) of (4.23) (Cyclic polytope) is possible.


