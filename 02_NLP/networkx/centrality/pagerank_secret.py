import privpy as pp
import pnumpy as pnp


def google_matrix(M, alpha = 0.85):
    import pnumpy as np
    N = len(M)
    if N == 0:
        return M
    #默认每个节点权重相同，即，1/n
    #vector初始化
    p = pp.farr(np.repeat(1.0 / N, N))

    #处理悬挂节点，即出度为0的节点
    temp1 = pnp.array([pnp.sum(M,axis=1) == 0])
    temp2 = pnp.full_like(M, 1/N)
    M = M+(temp1*temp2).trans()

    # #转移概率矩阵
    M = M/pnp.sum(M,axis=1)[:,None]

    return alpha * M + (1 - alpha) * p


def pagerank_numpy(G_Arry, alpha = 0.85):
    """

    :param G_Arry: （包含权重）邻接矩阵
    :param alpha: float, optional
      Damping parameter for PageRank, default=0.85.
    :param dangling: 悬挂节点dict, optional
      The outedges to be assigned to any "dangling" nodes, i.e., nodes without
      any outedges. The dict key is the node the outedge points to and the dict
      value is the weight of that outedge. By default, dangling nodes are given
      outedges according to the personalization vector (uniform if not
      specified) This must be selected to result in an irreducible transition
      matrix (see notes under google_matrix). It may be common to have the
      dangling dict to be the same as the personalization dict.
    :return: pagerank : dictionary
       Dictionary of nodes with PageRank as value.
    """
    M = pp.farr(google_matrix(G_Arry, alpha))
    eigenvalues, eigenvectors = pnp.linalg.linalg_sub.eig_vector_jacob(M.trans())
    ind = pnp.argmax(eigenvalues)
    # eigenvector of largest eigenvalue is at ind, normalized
    largest = pnp.array(eigenvectors[:, pp.back2plain(ind)]).flatten()
    norm = largest.sum()
    return largest/norm


def pagerank(G_Arry, alpha = 0.85, max_iter=100, tol=1.0e-6):
    M = pp.farr(google_matrix(G_Arry, alpha))
    # power iteration: make up to max_iter iterations
    N = len(M)
    if N == 0:
        return M
    x = pp.farr(pnp.repeat(1.0 / N, N))
    for _ in range(max_iter):
        xlast = x
        x = pnp.dot(xlast, M)
        # check convergence, l1 norm
        err = pnp.sum(pnp.abs(x-xlast))
        flag = err < N * tol
        flag_plain = pp.back2plain(flag)
        if flag_plain:
            break
    norm = x.sum()
    return x/norm


@pp.es
def main():
    ADJ = pp.farr([[0, 1, 2, 0], [2, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    res1 = pagerank_numpy(ADJ)
    pp.reveal(res1, "cipher://ds04/res1")
    res2 = pagerank(ADJ,max_iter=300,tol=1.0e-7)
    pp.reveal(res2,"cipher://ds04/res2")
pp.run()