
class Node:
        
        def __init__(self):
            self.color = None
            pass
        
        def setColor(self, color):
            self.color = color
        
        def getColor(self):
            return self.color

class Graph:
    
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges

    def get_neighbors(self, node):
        return self.edges[node]

class Problem:
    
    def __init__(self, countries, neighbors, colors):
        self.countries = countries
        self.neighbors = neighbors
        self.colors = colors
        
    def getColors(self):
        return self.colors
    
    def getCountries(self):
        return self.countries
    

class Solver:
    
    def __init__(self, problem):
        self.problem = problem
        
    def solve(self):
        countries = self.problem.getCountries()
        costraints = []
        for i in range(len(countries)):
            for j in range(len(countries)):
                if countries[i] in self.problem.neighbors[countries[j]]:
                    costraints.append((i, j))  
        graph = Graph(self.problem.getCountries(), costraints)