


def clone(list):
    new_list = []
    for i in range(len(list)):
        new_list.append(list[i])
    return new_list

def test(list):
    s = set()
    for i in range(len(list)):
        for j in range(i + 1, len(list)):
            s.add(abs(list[i] - list[j]))
    return len(s) == N

def find_sol(list):
    
    if len(list) == (N/2) + 1:
        return list
    
    else:
    
        candidates = []
        
        for i in range(3, len(list)):
            
            c = clone(list)
            c.remove(c[i])
            if test(c):
                candidates.append(c)
        
        if len(candidates) == 0:
            return list

        else:
            min_len = N
            min_sol = []
            for c in candidates:
                sol = find_sol(c)
                if len(sol) < min_len:
                    min_len = len(sol)
                    min_sol = sol
            if min_len < len(c):
                return min_sol
            else:
                return c
                
                
                
        
        
    
            


N = 30

list = []
minLength = N 

list.append(0)
list.append(1)
list.append(N)

best_sol = []
best_sol.append(clone(list))

for i in range(2, N):
    list.append(i)

print(find_sol(list))
        
        
