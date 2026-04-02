
import copy
from random import shuffle
from ordered_set import OrderedSet
from matplotlib.pylab import sort


class Vertex:
    def __init__(self, id):
        self.id = id
        self.neighbors = set()

    def add_neighbor(self, neighbor):
        self.neighbors.add(neighbor)

    def __repr__(self):
        return str(self.id)
    
    
class Graph:
    def __init__(self):
        self.vertices = {}
        self.edges = set()
        
    def add_vertex(self, id):
        if id not in self.vertices:
            self.vertices[id] = Vertex(id)
    
    def add_edge(self, id1, id2):
        self.add_vertex(id1)
        self.add_vertex(id2)
        self.vertices[id1].add_neighbor(id2)
        self.vertices[id2].add_neighbor(id1)
        self.edges.add((id1, id2))

    def get_neighbors(self, id):
        if id in self.vertices:
            return self.vertices[id].neighbors
        return None

    def get_edges(self):
        edges = set()
        for e in self.edges:
            edges.add(e)
            edges.add((e[1], e[0]))
        return edges

    def get_vertices(self):
        return self.vertices.values()
    
    def remove_vertex(self, id):
        if id in self.vertices:
            for neighbor in list(self.vertices[id].neighbors):
                self.vertices[neighbor].neighbors.remove(id)
                self.edges.discard((id, neighbor))
            del self.vertices[id]



def triangulate(graph):
    g = copy.deepcopy(graph)
    done = []
    vertices = list(g.get_vertices())
    vertices.sort(key=lambda x: len(x.neighbors))
    for v in vertices:
        for i in v.neighbors:
            for j in v.neighbors:
                if i != j and i not in done and j not in done:
                    if i not in g.get_neighbors(j):
                        g.add_edge(i, j)
        done.append(v.id)
    
    return g
        
    


if __name__ == "__main__":
    g = Graph()
    N = 8
    vertexes = [n for n in range(1, N)]
    #shuffle(vertexes)
    for vertex in vertexes:
        g.add_vertex(vertex)
    g.add_edge(1, 2)
    g.add_edge(1, 6)
    g.add_edge(2, 8)
    g.add_edge(3, 5)
    g.add_edge(3, 7)
    g.add_edge(4, 5)
    g.add_edge(4, 7)
    g.add_edge(5, 6)
    g.add_edge(7, 8)
    
    print("Before triangulation:")
    l = []
    for edge in g.get_edges():
        l.append(edge)
        l.sort(key=lambda x: x[0])
    print(l)
    g = triangulate(g)
    print("Vertices:")
    for vertex in g.get_vertices():
        print(vertex.id, end=" ")
    print("\nAfter triangulation:")
    for edge in g.get_edges():
        if edge not in l:
            print(edge)
            
print((2,1) in g.get_edges())