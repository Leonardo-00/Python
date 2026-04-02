from collections import defaultdict
import random
from time import sleep
import numpy as np
import time

class Tile:
    
    def __init__(self, x, y, n):
        self.x = x
        self.y = y
        self.number = n
    
    def __str__(self):
        return "[" + str(self.number) + "]"
        
class Actions:
    def __init__(self, state):
        self.actions = []
        for i in range(0, len(state.tiles)):
            a = []
            for j in range(0, len(state.tiles[i])):
                b = []
                if i > 0 and state.tiles[i-1][j] != None:
                    b.append("U")
                if i < len(state.tiles)-1 and state.tiles[i+1][j] != None:
                    b.append("D")
                if j > 0 and state.tiles[i][j-1] != None:
                    b.append("L")
                if j < len(state.tiles[i])-1 and state.tiles[i][j+1] != None:
                    b.append("R")
                a.append(b)
            self.actions.append(a)
    
    def getActions(self, x, y):
        return self.actions[x][y]


class State:
        
        def __init__(self, tiles):
            self.tiles = tiles
            d = 0
            k = len(tiles)
            for i in range(0, k):
                for j in range(0, k):
                    n = tiles[i][j]
                    if n == 0:
                        self.x = i
                        self.y = j
                    else:
                        x = n//k
                        y = n % k
                        if y == 0:
                            x -= 1
                            y = k-1
                        else:
                            y -= 1
                        d += abs(x- i) + abs(y - j)
            self.distance = d
            
        def clone(self):
            tiles = []
            for i in range(0, len(self.tiles)):
                a = []
                for j in range(0, len(self.tiles[i])):
                    a.append(self.tiles[i][j])
                tiles.append(a)
            s = State(tiles)
            return s
        
        def __str__(self):
            s =str(self.distance) + "\n"
            for i in range(0, len(self.tiles)):
                for j in range(0, len(self.tiles[i])):
                    s += "[" + self.tiles[i][j].__str__() + "]"
                s += "\n"
            return s
        
        def randomInitialState(k, border_done):
            tiles = [[0 for _ in range(k)] for _ in range(k)]
            nums = [x for x in range(1, pow(k,2))]
            nums.append(0)
            if border_done:
                for i in range(0, k):
                    tiles[0][i] = nums.pop(0)
                for i in range(1, k):
                    n = 1+k*i
                    nums.remove(n)
                    tiles[i][0] = n
                start = 1
            else:
                start = 0
            random.shuffle(nums)
            for i in range(start, k):
                for j in range(start, k):
                    if tiles[i][j] == 0:
                        tiles[i][j] = nums.pop()
            return State(tiles)

class KsquaredPuzzle:
    
    def __init__(self, k, border_done = False):
        self.k = k
        self.initial_state = State.randomInitialState(k, border_done)
        self.actions = Actions(self.initial_state)
        self.solvable = self.isSolvable(self.initial_state.tiles)
        
    def isSolvable(self, tiles):
        inversions = 0
        for i in range(0, len(tiles)):
            for j in range(0, len(tiles[i])):
                for m in range(i, len(tiles)):
                    n = 0
                    if m == i:
                        n = j+1
                    for l in range(n, len(tiles[m])):
                        if tiles[i][j] > tiles[m][l]:
                            inversions += 1
        if self.k % 2 == 0:
            if self.initial_state.x % 2 != 0:   
                return inversions % 2 != 0
            else:
                return inversions % 2 == 0
        else:
            return inversions % 2 == 0
        
    def result(self, state, action):
        s = state.clone()
        if action == "U":
            [x, y] = [s.x-1, s.y]
        elif action == "D":
            [x, y] = [s.x+1, s.y]
        elif action == "L":
            [x, y] = [s.x, s.y-1]
        elif action == "R":
            [x, y] = [s.x, s.y+1]
        
        tmp = s.tiles[x][y]
        s.tiles[x][y] = 0
        s.tiles[s.x][s.y] = tmp
        s.x = x
        s.y = y
        return s
    
    def goalTest(self, state):
        return state.distance == 0

class Node:
    def __init__(self, state, parent, action):
        self.state = state
        self.parent = parent
        self.action = action
        self.sequence = ""
        if parent == None:
            self.path = ""
            self.sequence = state.__str__()
        else:
            if parent.path == "":
                self.path = action
            else:
                self.path = parent.path + " -> " + action
            self.sequence = parent.sequence + state.__str__()
        if parent == None:
            self.path_cost = 0
            self.depth = 0
        else:
            self.path_cost = parent.path_cost + 1
            self.depth = parent.depth + 1        
    
    def __str__(self):
        return self.state.__str__() + str(self.path_cost) + "\n"
        
class Frontier:
    def __init__(self, order):
        self.__queue = []
        self.order = order
        
    
    def insert(self, node):
        self.__queue.append(node)
        if self.order == "BFS":
            self.__queue.sort(key=lambda x: x.depth)
        elif self.order == "DFS":
            self.__queue.sort(key=lambda x: x.depth, reverse=True)
        elif self.order == "UCS":
            self.__queue.sort(key=lambda x: x.path_cost)
        elif self.order == "A*":
            self.__queue.sort(key=lambda x: x.path_cost + x.state.distance)
    
    def get(self):
        return self.__queue.pop(0)
    
    def empty(self):
        return len(self.__queue) == 0
    

class Solver:
        
        def __init__(self, problem, order):
            self.problem = problem
            self.frontier = Frontier(order)
            self.explored = defaultdict(int)
        
        def solve(self):
            
            actions = self.problem.actions.getActions(self.problem.initial_state.x, self.problem.initial_state.y)
            self.__explored(self.problem.initial_state, 0)
            root = Node(self.problem.initial_state, None, None)
            if problem.goalTest(root.state):
                return root
            for a in actions:
                s = self.problem.result(self.problem.initial_state, a)
                self.frontier.insert(Node(s, root, a))
            while not self.frontier.empty():
                print(len(self.explored))
                node = self.frontier.get()
                if problem.goalTest(root.state):
                    return node
                self.__explored(node.state, node.depth)
                actions = self.problem.actions.getActions(node.state.x, node.state.y)
                for a in actions:
                    s = self.problem.result(node.state, a)
                    if self.__notExplored(s, node.depth+1):
                        self.frontier.insert(Node(s, node, a))
            return None
        
        def __explored(self, state, depth):
            self.explored[state.__str__()] = depth
        
        def __notExplored(self, state, depth):
            if self.explored.__contains__(state.__str__()) == False or self.explored[state.__str__()] > depth:
                self.explored[state.__str__()] = depth
                return True
            return False


k = 4
problem = KsquaredPuzzle(k, True)
solver = Solver(problem, "A*")
s = problem.initial_state

t = time.time()
node = solver.solve()
print("Time: " + str(time.time() - t))
if node != None:
    sol = node.state
    print(node.path)
    print(s.__str__())
else:
    print("No solution found")

print(s.__str__())
print("Is solvable: " + str(problem.solvable))
