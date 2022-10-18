import pagerank
import numpy as np
import networkx_yy as nx


A = np.array([[0,1,2],[1,0,3],[1,3,0]])

# G = nx.DiGraph()
# for i in range(len(A)):
# 	for j in range(len(A)):
# 		G.add_edge(i,j,weight=A[i,j])


# graph = nx.from_numpy_array(A)
# print(nx.adj_matrix(G))
# print("特征向量：",pagerank.pagerank1(A))
# print("幂迭代：",pagerank.pagerank1(A))
# print("************************")
# print(nx.pagerank(G).values())
# print(nx.pagerank_numpy(G).values())
# print(nx.pagerank_scipy(G).values())
N = len(A)
x = np.repeat(1.0 / N, N)
for _ in range(100):
	xlast = x
	x = np.dot(xlast, A)
	# check convergence, l1 norm
	err = sum([np.abs(x[n] - xlast[n]) for n in range(len(x))])
norm = x.sum()
print(x/norm)

eigenvalues, eigenvectors = np.linalg.eig(A.T)
ind = np.argmax(eigenvalues)
largest = np.array(eigenvectors[:, ind]).flatten().real
norm = float(largest.sum())
print(largest/norm)
