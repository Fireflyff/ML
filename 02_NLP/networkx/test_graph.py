from Graph import Graph
from Graph import Shortest_path
from Graph import Core
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np

# G = nx.read_gml("/Users/yingying/Downloads/dolphins/dolphins.gml")
# G.remove_node()
# G = np.array(nx.adjacency_matrix(G).todense())
# S = Shortest_path(adj_graph)
# print(S.shortest_path()[7][49])


G = nx.DiGraph()
# G.add_node()
# G.nodes()
# edgelist = [
#     (0,1),
#     (0,2),
#     # (0,3),
#     (3,2),
#     (3,1),
#     # (1,2),
#     (1,8),
#     (8,4),
#     (4,5),
#     # (4,6),
#     (4,7),
#     (6,7),
#     (6,5),
#     # (5,7),
#     (9,0),
#     (9,1),
#     (9,2),
#     (9,3),
#     (10,4),
#     (10,5),
#     (10,6),
#     (10,7)
# ]
edgelist = [(0,1),(0,2),(0,3),(3,4),(2,5),(5,6),(4,6)]

# edgelist = [
#     (0,1),(0,2),(0,3),
#     (1,4),(1,5),(1,6),
#     (2,7),(2,8),(2,9),
#     (3,10),(3,11),(3,12),
#     (4,13),(4,14),(4,16),
#     (5,19),(5,17),(5,18),
#     (6,21),(6,22),(6,23),
#     (7,24),(7,25),(7,26),
#     (8,27),(8,28),(8,29),
#     (9,30),(9,31),(9,32),
#     (10,33),(10,34),(10,35),
#     (11,36),(11,37),(11,38),
#     (12,39),(12,40),(12,41)
# ]

# edgelist = [
#     (0,1),
#     (0,2),
#     (1,2),
#     (3,4),
#     (3,5),
#     (3,6),
#     (6,5),
#     (6,4),
#     (4,5),
#     (7,8),
#     (7,9),
#     (7,10),
#     (7,11),
#     (8,9),
#     (8,10),
#     (8,11),
#     (9,10),
#     (9,11),
#     (10,11)
# ]

G.add_edges_from(edgelist)
# G.remove_edge()
# G.add_edges_from()
# k = 3
GG = np.array(nx.adjacency_matrix(G).todense())
print(GG)
# P = Shortest_path(GG)
# print(P.shortest_path())

# A = Core(GG)
# GG = A.k_core(k)
# GG = nx.from_numpy_array(GG)
# # GG = nx.k_core(G,k)
# # G = np.array(nx.adjacency_matrix(GG).todense())
# # print(len(GG.nodes()))
# # print(len(GG.edges()))
# pos = nx.fruchterman_reingold_layout(G)
# nx.draw_networkx(G, pos, cmap=plt.get_cmap('tab20'), node_size=200, edge_color='gray', with_labels=True)
# plt.show()
print(nx.shortest_path(G))
# nx.closeness_centrality()
# nx.k_shell()
# nx.from_numpy_array()
# G.add_edge()

