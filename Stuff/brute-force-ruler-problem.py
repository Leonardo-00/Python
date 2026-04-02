from itertools import combinations
import math

def subset(lista, k):
    
    tutti_i_subset = []
    N = len(lista)
    for r in range(k-1, k+1):
        subset_r = combinations(lista, r)
        tutti_i_subset.extend(list(sub) for sub in subset_r)
    return tutti_i_subset


def clone(list):
    new_list = []
    for i in range(len(list)):
        new_list.append(list[i])
    return new_list

def test(list, n):
    s = set()
    for i in range(len(list)):
        for j in range(i + 1, len(list)):
            s.add(abs(list[i] - list[j]))
    return len(s) == n

def sol_length(n):
    
    k = math.ceil((1 + math.sqrt(1 + 8 * n)) / 2)
    
    l = []
    sols = []
    l.append(0)

    l1 = []

    for j in range(1, n+1):
        l1.append(j)
        
    sub = subset(l1, k)

    for list in sub:
        candidate = clone(l)
        candidate.extend(list)
        if test(candidate, n):
            candidate.sort()
            sols.append(candidate)


    sols = sorted(sols, key=len)
    if len(sols) == 0:
        return None
    else:
        return len(sols[0])
    
def sol(n):
    
    k = math.ceil((1 + math.sqrt(1 + 8 * n)) / 2)
    
    l = []
    sols = []
    l.append(0)

    l1 = []

    for j in range(1, n+1):
        l1.append(j)
        
    sub = subset(l1, k)

    for list in sub:
        candidate = clone(l)
        candidate.extend(list)
        if test(candidate, n):
            candidate.sort()
            sols.append(candidate)


    sols = sorted(sols, key=len)
    if len(sols) == 0:
        return None
    else:
        return sols[0]
    

lens = []
N = 30
"""
for i in range(5, N+1):
    lens.append((i, sol_length(i)))
print(lens)
"""
print(sol(N))



