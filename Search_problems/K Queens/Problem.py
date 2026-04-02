import random


def getAttackedQueens(solution):
    queens = solution.queens if hasattr(solution, "queens") else solution
    attacked = set()
    k = len(queens)
    for i in range(k):
        for j in range(i+1, k):
            if queens[i] == queens[j] or abs(i-j) == abs(queens[i]-queens[j]):
                attacked.add(i)
                attacked.add(j)
    return attacked


def buildBoard(solution, colorAttacks=False):
    queens = solution.queens if hasattr(solution, "queens") else solution
    attacked = getAttackedQueens(queens)
    k = len(queens)
    s = ""
    for row in range(1, k+1):
        for col in range(k):
            if queens[col] == row:
                if colorAttacks and col in attacked:
                    s += "\033[91m[X]\033[0m"
                else:
                    s += "[X]"
            else:
                s += "[ ]"
        s += "\n"
    return s


class State:
    def __init__(self, k, pos):
        self.k = k
        self.queens = []*k
        for i in range(k):
            self.queens.append(pos[i])
        self.score = self.__getScore()
    
    def __str__(self):
        s = "[" + str(self.queens[0]) + "]"
        for i in range(1, self.k):
            s += " [" + str(self.queens[i]) + "]"
        return s
    
    def showGrid(self, colorAttacks=False):
        return buildBoard(self.queens, colorAttacks=colorAttacks)
    
    def __getScore(self):
        score = 0
        for i in range(self.k):
            for j in range(i+1, self.k):
                if self.queens[i] == self.queens[j]:
                    score += 1
                if abs(i-j) == abs(self.queens[i]-self.queens[j]):
                    score += 1
        return score


class Problem:
    def __init__(self, k):
        self.k = k
        pos = []
        for i in range(k):
            pos.append(random.randint(1, k))
        self.initial = State(k, pos)
    
    def getActions(self, state):
        actions = []
        for i in range(state.k):
            for j in range(1, state.k+1):
                if j != state.queens[i]:
                    actions.append((i,j))
        return actions
    
    def result(self, state, action):
        pos = []
        for i in range(state.k):
            pos.append(state.queens[i])
        pos[action[0]] = action[1]
        return State(state.k, pos)

    def goalTest(self, state):
        for i in range(state.k):
            for j in range(i+1, state.k):
                if state.queens[i] == state.queens[j]:
                    return False
                if abs(i-j) == abs(state.queens[i]-state.queens[j]):
                    return False
        return True
  
class Node:
    
    def __init__(self, state):
        self.state = state
        self.score = state.score


class Queue:
    
    def __init__(self):
        self.queue = []
        
    def push(self, node):
        self.queue.append(node)
        self.queue.sort(key=lambda x: x.score)
        
    def pop(self):
        return self.queue.pop(0)
        
 
class Solver:
    def __init__(self, problem):
        self.problem = problem
        self.visitedStates = []
        
    def solve(self):
        
        current = Node(self.problem.initial)
        while True:
            actions = self.problem.getActions(current.state)
            neighbors = Queue()
            for i in range(len(actions)):
                s = self.problem.result(current.state, actions[i])
                if self.__notVisited(s):
                    neighbors.push(Node(s))
                    neigh = neighbors.pop()
                    if neigh.score > current.score:
                        return current.state
                    if self.problem.goalTest(neigh.state):
                        return neigh.state
                    current = neigh

    
    def __notVisited(self, state):
        for i in range(len(self.visitedStates)):
            for j in range(self.problem.k):
                visited = True
                if self.visitedStates[i].queens[j] != state.queens[j]:
                    visited = False
                    break
            if visited:
                return False
        return True
    
class SimulatedAnnealing(Solver):
    def __init__(self, problem):
        self.problem = problem
        self.visitedStates = []
        
    def solve(self):
        current = Node(self.problem.initial)
        T = 2.0
        T_min = 0.00001
        alpha = 0.9
        while T > T_min:
            i = 1
            while i <= 100:
                actions = self.problem.getActions(current.state)
                next = random.choice(actions)
                next_state = self.problem.result(current.state, next)
                E = current.state.score - next_state.score
                if E > 0:
                    current = Node(next_state)
                else:
                    if random.random() < pow(2.71828, E/T):
                        current = Node(next_state)
                if self.problem.goalTest(current.state):
                    return current.state
                i += 1
            T = T * alpha
        return current.state


if __name__ == "__main__":
    problem = Problem(7)
    s = SimulatedAnnealing(problem)
    solution = s.solve()
    print("Score: " + str(solution.score))
    print(solution.showGrid(colorAttacks=True))