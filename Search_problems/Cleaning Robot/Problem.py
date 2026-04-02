from collections import defaultdict
import random
from time import sleep, time
import numpy as np
from termcolor import colored

class Tile:
    
    def __init__(self, x, y, clean):
        self.x = x
        self.y = y
        self.clean = clean
    
    def __str__(self):
        if(self.clean):
            return "[ ]"
        else:
            return "[X]"
    
    def clean(self):
        self.clean = True
        
class Actions:
    def __init__(self, state):
        self.actions = []
        for i in range(0, len(state.tiles)):
            a = []
            for j in range(0, len(state.tiles[i])):
                b = []
                if state.tiles[i][j] != None and state.tiles[i][j].clean == False:
                    b.append("C")
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
        
        def __init__(self, tiles, x, y):
            self.tiles = tiles
            self.x = x
            self.y = y
            d = 0
            for i in range(0, len(tiles)):
                for j in range(0, len(tiles[i])):
                    if tiles[i][j] != None and tiles[i][j].clean == False:
                        d += 1
            self.dirty = d
            
        def clone(self):
            tiles = []
            for i in range(0, len(self.tiles)):
                a = []
                for j in range(0, len(self.tiles[i])):
                    if self.tiles[i][j] != None:
                        a.append(Tile(i, j, self.tiles[i][j].clean))
                    else:
                        a.append(None)
                tiles.append(a)
            s = State(tiles, self.x, self.y)
            s.dirty = self.dirty
            return s
        
        def __str__(self):
            s = "Dirty tiles: " + str(self.dirty) + "\n"
            for i in range(0, len(self.tiles)):
                for j in range(0, len(self.tiles[i])):
                    if self.tiles[i][j] == None:
                        s += "|||"
                    else:
                        if i == self.x and j == self.y:
                            s += f"{'\033[92m'}"
                        if self.tiles[i][j].clean:
                            s += "[ ]"
                        else:
                            s += "[X]"
                        if i == self.x and j == self.y:
                            s += "\033[0m"
                s += "\n"
            return s
        
        def clean(self):
            if self.tiles[self.x][self.y].clean == False:
                self.dirty -= 1
            self.tiles[self.x][self.y].clean = True
        
        def randomInitialState(k, p):
            tiles = randomBoard(k, p)
            x = random.randint(0, k-1)
            y = random.randint(0, k-1)
            while tiles[x][y] == None:
                x = random.randint(0, k-1)
                y = random.randint(0, k-1)
            s = State(tiles, x, y)
            return s
            
def randomBoard(k, p):
    tiles = []
    for i in range(0, k):
        a = []
        for j in range(0, k):
            q = random.random()
            if(q > p):
                q = random.random()
                if(q < 1/3):
                    a.append(Tile(i, j, False))
                else:
                    a.append(Tile(i, j, True))
            else:
                a.append(None)
        tiles.append(a)
    return tiles


class VacuumProblem:        
    
    def __init__(self, k, p):
        self.initial_state = State.randomInitialState(k, p)
        self.actions = Actions(self.initial_state)
        
    def result(self, state, action):
        s = state.clone()
        if action == "U":
            s.x -= 1
        elif action == "D":
            s.x += 1
        elif action == "L":
            s.y -= 1
        elif action == "R":
            s.y += 1
        elif action == "C":
            s.clean()
        return s

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
    
    def getPath(self):
        if self.parent == None:
            return [self]
        
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
            self.__queue.sort(key=lambda x: x.path_cost + x.state.dirty)
    
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
            if root.state.dirty == 0:
                return root
            for a in actions:
                s = self.problem.result(self.problem.initial_state, a)
                self.frontier.insert(Node(s, root, a))
            while not self.frontier.empty():
                node = self.frontier.get()
                if node.state.dirty == 0:
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


def checkVisitedStates(state, visitedStates):
    for s in visitedStates:
        if s.__str__() == state.__str__():
            repeatedStates += 1
            break
    visitedStates.append(state.__str__())
