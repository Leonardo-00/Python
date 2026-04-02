import math
import random


class Problem:
    
    def __init__(self, goal):
        self.start = 4
        self.goal = goal
    
    def getActions(self, state):
        a = []
        a.append("sqrt")
        if state.__class__ == int and state < 198:
            a.append("fact")
        else:
            a.append("floor")
        return a
    
    def isGoal(self, state):
        return state == self.goal
    
    def result(self, state, action):
        if action == "fact":
            return math.factorial(state)
        elif action == "floor":
            return math.floor(state)
        elif action == "sqrt":
            return math.sqrt(state)
        else:
            return state
        
class Node:
    
    def __init__(self, state, parent = None, action = None):
        self.state = state
        self.parent = parent
        self.action = action
        if parent == None:
            self.path = "4"
            self.depth = 0
            self.path_cost = 0
        else:
            self.path = action + "(" + parent.path + ")"
            self.depth = parent.depth + 1
            self.path_cost = parent.path_cost + 1

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
        self.explored = {}
    
    def solve(self) -> Node:
        self.frontier.insert(Node(self.problem.start))
        while self.frontier.empty() == False:
            node = self.frontier.get()
            state = node.state
            if self.problem.isGoal(state):
                return node
            if state in self.explored:
                continue
            self.explored[state] = True
            for a in self.problem.getActions(state):
                if self.explored.__contains__(self.problem.result(state, a)) == False:
                    self.frontier.insert(Node(self.problem.result(state, a), node, a))
        return None
    

n = 0
for i in range(1, 100):
    goal = i
    problem = Problem(goal)
    solver = Solver(problem, "BFS")
    if solver.solve() != None:
        n += 1
print(n)
