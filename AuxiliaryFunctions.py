"""
This file is devoted to store the auxiliary functions that are necessary for computing effectively.
"""
import itertools
# sample_word = 'abc'
reversed_gen_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'C', 4: 'B', 5: 'A'}
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
                nlw[i] = reversed_gen_dict[perm[0]]
            if word[i] == 'A':
                nlw[i] = reversed_gen_dict[5 - perm[0]]
            if word[i] == 'b':
                nlw[i] = reversed_gen_dict[perm[1]]
            if word[i] == 'B':
                nlw[i] = reversed_gen_dict[5 - perm[1]]
            if word[i] == 'c':
                nlw[i] = reversed_gen_dict[perm[2]]
            if word[i] == 'C':
                nlw[i] = reversed_gen_dict[5 - perm[2]]
        res.add(''.join(nlw))    
    return res

# print(get_all_renamed_words(sample_word))