from AuxiliaryFunctions import get_all_renamed_words, is_word_regular
import gurobipy as gp
import math
from gurobipy import GRB
import numpy as np
import itertools
import networkx
# We work over the free group of rank 3.
generator = ["a", "b", "c", "A", "B", "C"]
n = len(generator)
sample_list_of_words = ["abcABcc",]
def get_length_two_subwords(list_of_words: list) -> list:
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
    res = set()
    C = A.union(B)
    for w in C:
        S = get_all_renamed_words(w)
        res = res.union(S)
    return res.union(C)
def get_edge_pairing_of_single_word(word: str, w_index: int) -> dict:
    dict_of_edge_pairing = {"a": [], "b": [], "c": []}
    l = get_length_two_subwords([word,])
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
def get_whitehead_graph_with_edge_pairing(list_of_words: list) -> (np.ndarray, dict):
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
def get_adjacent_edges(incidence_matrix: np.ndarray) -> dict:
    res = {'a': [], 'A': [], 'b': [], 'B': [], 'c': [], 'C': []}
    for i in range(np.shape(incidence_matrix)[-1]):
        if incidence_matrix[:, i][0] == 1:
            res['a'].append(i)
        if incidence_matrix[:, i][1] == 1:
            res['A'].append(i)
        if incidence_matrix[:, i][2] == 1:
            res['b'].append(i)
        if incidence_matrix[:, i][3] == 1:
            res['B'].append(i)
        if incidence_matrix[:, i][4] == 1:
            res['c'].append(i)
        if incidence_matrix[:, i][5] == 1:
            res['C'].append(i)
    return res
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
                word_adder = []
                for j in range(len(v)):
                    if v[j] == 1:
                        vertices_of_j = "".join(reversed_gen_dict[i] for i in range(6) if incidence_matrix[i, j] == 1)
                        # names are irrelevant
                        word_adder.append(vertices_of_j) 
                all_c[v] = word_adder
    return all_c

# Calculatable now.

def get_set_of_all_good_words_of_length_at_most(length : int) -> set:
    # minimal_diskbusting_cyclically_reduced
    res = set()
    for i in range(1, length + 1):
        for j in itertools.product(generator, repeat = i):
            word = "".join(j)
            if not is_word_regular(word):
                if is_cyclically_reduced(word) and is_minimal_and_diskbusting([word,]) and res.isdisjoint(get_redundant_words(word)):
                    res.add(word)
    return res
def get_set_of_all_good_words_of_length(length : int) -> set:
    # minimal_diskbusting_cyclically_reduced
    res = set()
    i = length
    for j in itertools.product(generator, repeat = i):
        word = "".join(j)
        if not is_word_regular(word):
            if is_cyclically_reduced(word) and is_minimal_and_diskbusting([word,]) and res.isdisjoint(get_redundant_words(word)):
                res.add(word)
    return res

death = []
def get_gcd_of_list(l: list) -> int:
    res = l[0]
    for i in l:
        res = math.gcd(res, i)
    return res

w_length = 11
checklist3 = ['aabcaBcbcbC', 'aabcaBCBcbC', 'aabcbcbACbC', 'aabcbACBCbC', 'aabacbccabb', 'aaabcbcBcBC', 'aaaabaccbbc', 'aabcbAcbCBc', 'aabbaccbabc', 'aabcBCBcbAc', 'aaaabcBcbAC', 'aaabcbCbAAc', 'aabcbCBcbAc', 'aaabcaaBcbC', 'aaabcbcBcbC', 'aaabcbAAcBc', 'aabbaacbbcc', 'aabcBcbCbAc', 'aaabcbcBCbC', 'aabcbCbCbAC', 'aabcBcbcbAC', 'aabcaBCBCbC', 'aabcbAcBCbc', 'aabcBCbCbAc', 'aaaabcaBcBC', 'aaabbabbccc', 'aaabbaccbbc', 'aaabcBCbCBc', 'aababccbbac', 'aaabbccbccb', 'aaabcBCbAAC', 'aabcaBCBcBC', 'aabcbACbcBC', 'aabcbAcBCBc', 'aaaabcbAcBc', 'aabcaBCbCbC', 'aabcaBCbCBC', 'aabcbcbCbAc', 'aaabbaaccbc', 'aabcbACBcBC', 'aabcaBcBCBC', 'aaabcbCbCBC', 'aaaaabbcccb', 'aabcBcBcbAc', 'aaabbbaaccc', 'aabcbACbCBC', 'aababbacbcc', 'aaaabcBcbAc', 'aabcbCbcbAC', 'aabcbACBcbC', 'aabcBcBCbAc', 'aaabbacccbb', 'aabcbcBcbAC', 'aabacbabbcc', 'aaababcccbb', 'aaaabbccacb', 'aabcaBcBcBC', 'aabcbAcbCbc', 'aabbacabccb', 'aabcBcbcbAc', 'aaababbcccb', 'aaabcaaBcBC', 'aabcBCbcbAc', 'aaabcbCbcBC', 'aabaccabbcb', 'aabcBcBcbAC', 'aabbacbabcc', 'aaaaabbbccc', 'aaaabbcaccb', 'aaabcbCbCBc', 'aaaabbbbccc', 'aabcbCBCbAc', 'aaaabbaccbc', 'aababcbbacc', 'aababbaccbc', 'aaabacbbccc', 'aaabcbCbcbC', 'aabcBCbcbAC', 'aaaabbcccbb', 'aababacbbcc', 'aabcbcBCbAc', 'aabcBcaBcBC', 'aaabbccaacb', 'aabcBcBCbAC', 'aaabcbcbCbC', 'aabcbCBcbAC', 'aaabcaaBCbC', 'aabcBCBcbAC', 'aaabcBcbcBC', 'aaabcbCbcBc', 'aabcBCbCbAC', 'aaabbcaaccb', 'aaabcBcbCBc', 'aabacbbcacc', 'aabcBcbCbAC', 'aaabbaccbcc', 'aababaccbbc', 'aabcbCaBcbc', 'aaabbbacccb', 'aaabaccbbbc', 'aabcBcaBcbC', 'aaabcBCbcBC', 'aaabcBcBCbc', 'aabacabccbb', 'aaabaaccbbc', 'aabcaBcBCbC', 'aabcbAcbcBc', 'aaabcbCbAAC', 'aabcbCaBCBc', 'aaabbcaccbb', 'aaaabbbaccc', 'aabcbCBCbAC', 'aaabaacbbcc', 'aaabcBcbCbc', 'aaaabcaBcbC', 'aaabbbaccbc', 'aabcbACbcbC', 'aaabacccbbc', 'aabacacbbcc', 'aabcaBcbcBC', 'aaaabbbcccb', 'aabcbcaBCbC', 'aabaaccbbcc', 'aababcaccbb', 'aaabbaacbcc', 'aabcbAcBcbc', 'aabcbcBCbAC', 'aabacbbccbc', 'aaabcbCBcBC', 'aaaabcBCbAc', 'aabcbCaBCbc', 'aaabacccbbb', 'aaabbabcccb', 'aaabbccaccb', 'aaaabbacbcc', 'aaababbbccc', 'aabcBCaBCbc', 'aabacbbabcc', 'aaabccbbccb', 'aabcbcaBcBC', 'aabacbbccab', 'aaaabbccccb', 'aabcabbaccb', 'aaabbbacbcc', 'aaabcbCBcbC', 'aabcaBcBcbC', 'aaabcbCbCbc', 'aaabcBcbcbC', 'aabcbcbCbAC', 'aaaabacbbcc', 'aababccacbb', 'aaabbacbbcc', 'aabcbcbAcBc', 'aabacabbccb', 'aaaabcBCbAC', 'aaabcBcbAAc', 'aabcaBcbCBC', 'aaabaccbbcc', 'aabcaBCbcBC', 'aabcbcBcbAc', 'aaaabcbCbAC', 'aaabbccbbcc', 'aaabcBcbAAC', 'aababbcbacc', 'aabcBCBCbAc', 'aabcbCbCbAc', 'aaaabcaBCbC', 'aabcbCbcbAc', 'aabcBCBCbAC', 'aaabcBCbAAc', 'aabcbcaBcbC', 'aabcaBCbcbC', 'aabaccbabbc', 'aaaabcbCbAc', 'aabcaBcbCbC', 'aaabacbbbcc']
sol_dict = dict()
for w in checklist3:
    l = [w,]
    A_G, edge_pairing = get_whitehead_graph_with_edge_pairing(l)
    cycles_dict = get_all_cycles(A_G)
    cycles = list(cycles_dict.keys())
    cycle_mat = np.array(cycles)
    l = len(cycles)
    # gurobi model
    with gp.Env(empty = True) as env:
        env.setParam('LogToConsole', 0)
        # not printing logs to console.
        env.start()
        with gp.Model(env = env) as model:
            vars = model.addMVar(shape = (1, l+4), vtype = GRB.INTEGER, lb = 0)
            # l+1 is for a and A
            # l+2 is for b and B
            # l+3 is for c and C
# An imposed condition. Let's see if it actually works.
            model.addConstr(gp.quicksum(vars[0, j] for j in range(l) if sum(cycles[j]) >= 3) >= 1)
            for index in range(len(w)):
                model.addConstr(gp.quicksum(vars[0, j_index] for j_index in range(l) if cycles[j_index][index] == 1) == vars[0, l])                    
            # conditions for balance
            # change condition for balance: for each vertex in a, b, c, the number of occurrence of each pair of incident edges is the same. 
            adjacency = get_adjacent_edges(A_G)
            for gen in ["a", "b", "c"]:
                if gen == 'a':
                    v = l+1
                if gen == 'b':
                    v = l+2
                if gen == 'c':
                    v = l+3
                for (i, j) in itertools.combinations(adjacency[gen], 2):
                    model.addConstr(gp.quicksum(vars[0, j_index] for j_index in range(l) if cycles[j_index][i] == 1 and cycles[j_index][j] == 1) == vars[0, v])
                for (i, j) in itertools.combinations(adjacency[gen.upper()], 2):
                    model.addConstr(gp.quicksum(vars[0, j_index] for j_index in range(l) if cycles[j_index][i] == 1 and cycles[j_index][j] == 1) == vars[0, v])
                
                # for (i, j) in itertools.combinations(edge_pairing[gen], 2):
                #     model.addConstr(gp.quicksum(vars[0, k] for k in range(l) if cycles[k][i[0]] == 1 and cycles[k][j[0]] == 1) == gp.quicksum(vars[0, k] for k in range(len(cycles)) if cycles[k][i[1]] == 1 and cycles[k][j[1]] == 1))
            # calculations
            model.setObjective(gp.quicksum(vars[0, 0:l]), sense = GRB.MINIMIZE)
            model.Params.MIPFocus = 1
            # focus on finding feasible solutions
            model.optimize()
            try:
                sol = [int(i) for i in model.X[0:l]]
            except:
                death.append(w)
            else:
                dict_of_cycles_used = dict()
                for k in range(len(sol)):
                    if sol[k] > 0:
                        dict_of_cycles_used[(k,) + tuple(cycles_dict[cycles[k]])] = sol[k]
                sol_dict[w] = dict_of_cycles_used

with open(f'reduced_list_of_length_{w_length}_words_with_STRONG_imposed_condition.txt', 'w') as sourceFile:
    for i in sol_dict.keys():
        sourceFile.write(f'{i}, {sol_dict[i]}\n')
    sourceFile.close()
print(f'Funny jokes :D. {death}')


