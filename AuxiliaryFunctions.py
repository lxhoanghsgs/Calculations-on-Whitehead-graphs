"""
This file is devoted to store the auxiliary functions that are necessary for computing effectively.
"""
import itertools
import numpy as np
# import igraph
import networkx
# We work over the free group of rank 3.
generator = ["a", "b", "c", "A", "B", "C"]
gen_dict = {"a": 0, "A": 1, "b": 2, "B": 3, "c": 4, "C": 5}
reversed_gen_dict_aux = {0: 'a', 1: 'b', 2: 'c', 3: 'C', 4: 'B', 5: 'A'}
n = len(generator)
def get_length_two_subwords(list_of_words: list) -> list:
    list_of_length_two_subwords = []
    for i in list_of_words:
        for j in range(-1, len(i)-1):
            list_of_length_two_subwords.append(i[j] + i[j+1])
    return list_of_length_two_subwords

def is_cyclically_reduced(word: str) -> bool:
    return all(i not in get_length_two_subwords([word,]) for i in ["aA", "Aa", "bB", "Bb", "cC", "Cc"])

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

def get_inverse(word: str) -> str:
    inversed_word = list(word[len(word) - 1 - i] for i in range(len(word)))
    for i in range(len(inversed_word)):
        if inversed_word[i].islower():
            inversed_word[i] = inversed_word[i].upper()
        else:
            inversed_word[i] = inversed_word[i].lower()
    return "".join(inversed_word)

def check_first_chunk(word: str) -> bool:
    """
    Return 
    """
# Can optimize this a little bit more by removing certain words in the set of redundant words.
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
# sample_word = 'abc'
possible_permutation = [j for j in itertools.permutations(range(6), 3) if all(j[i] + j[i+1] != 5 for i in range(-1, 2))]
possible_permutation.remove((0, 1, 2))
def is_word_regular(word: str) -> bool:
    """
    Returns True if the numbers of occurences of 'a' and 'A', 'b' and 'B', 'c' and 'C' in word are equal.
    """
    n1 = word.count('a') + word.count('A')
    n2 = word.count('b') + word.count('B')
    n3 = word.count('c') + word.count('C')
    return any([n1 <= 2, n2 <= 2, n3 <= 2]) or all([n1 == n2, n1 == n3])  
def get_all_renamed_words(word: str) -> set:
    """
    Returns all words that are renamings of the original word.
    Example: get_renaming_of_a_word('abc') = {'bac', ...} (replace b by a, a by b)
    """
    res = {word,}
    for perm in possible_permutation:
        nlw = list(word)
        for i in range(len(word)):
            if word[i] == 'a':
                nlw[i] = reversed_gen_dict_aux[perm[0]]
            if word[i] == 'A':
                nlw[i] = reversed_gen_dict_aux[5 - perm[0]]
            if word[i] == 'b':
                nlw[i] = reversed_gen_dict_aux[perm[1]]
            if word[i] == 'B':
                nlw[i] = reversed_gen_dict_aux[5 - perm[1]]
            if word[i] == 'c':
                nlw[i] = reversed_gen_dict_aux[perm[2]]
            if word[i] == 'C':
                nlw[i] = reversed_gen_dict_aux[5 - perm[2]]
        res.add(''.join(nlw))    
    return res

def get_reduced_set_of_good_word(length: int) -> set:
    """
    Returns a non-redundant set of words of the given length in a somewhat better (faster) way.
    """
    res = set()
    for i in range(2, length - 1):
        for j in range(1, i):
            # The first j words are all 'a', the next i-j words are all 'b'.
            init_word = 'a' * j + 'b' * (i - j)
            possible_next_letter = {'a', 'A', 'c', 'C'}
            possible_last_letter = {'b', 'B', 'c', 'C'}
            for nl, ll in itertools.product(possible_next_letter, possible_last_letter):
                for middle_letters in itertools.product(reversed_gen_dict_aux.values(), repeat = length - i - 2):
                    # repeat = 0 in itertools.product is OK.
                    forming_word = init_word + nl + "".join(middle_letters) + ll
                    if not is_word_regular(forming_word):
                        if is_cyclically_reduced(forming_word):
                            if is_minimal_and_diskbusting([forming_word,]):
                                if res.isdisjoint(get_redundant_words(forming_word)):
                                    res.add(forming_word)            
                        # All conditions added
    return res

