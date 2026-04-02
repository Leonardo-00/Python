import networkx as nx
from networkx import is_chordal

l = [(1, 2),(1, 6),(2, 8),(3, 5),(3, 7),(4, 5),(4, 7),(5, 6),(7, 8)]
g = nx.Graph(l)

print(g.nodes())
print(g.edges())
print(is_chordal(g))
G = nx.complete_to_chordal_graph(g)[0]
for i in G.edges():
    if i not in g.edges():
        print(i)

print(g.has_edge(2,1))

def bron_kerbosch(graph, r, p, x, cliques):
    if not p and not x:
        cliques.append(r)
        return
    for v in list(p):
        bron_kerbosch(
            graph,
            r.union({v}),
            p.intersection(graph.neighbors(v)),
            x.intersection(graph.neighbors(v)),
            cliques
        )
        p.remove(v)
        x.add(v)

def find_maximal_cliques(graph):
    cliques = []
    bron_kerbosch(graph, set(), set(graph.nodes()), set(), cliques)
    return cliques

# Example usage
maximal_cliques = find_maximal_cliques(G)
print("Maximal cliques:", maximal_cliques)