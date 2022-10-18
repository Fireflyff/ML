import numpy as np
def google_matrix(M, alpha = 0.85):
    M.dtype = np.float
    N = len(M)
    if N == 0:
        return M
    #默认每个节点权重相同，即，1/n
    #vector初始化
    p = np.repeat(1.0 / N, N)

    #处理悬挂节点，即出度为0的节点
    dangling_nodes = np.where(M.sum(axis=1) == 0)[0]
    for node in dangling_nodes:
        M[node] = p

    #转移概率矩阵
    M = M/M.sum(axis=1)[:,None]
    return alpha * M + (1 - alpha) * p


def google_matrix1(M, alpha = 0.85):
    M.dtype = np.float
    N = len(M)
    if N == 0:
        return M
    #默认每个节点权重相同，即，1/n
    #vector初始化
    p = np.repeat(1.0 / N, N)

    #处理悬挂节点，即出度为0的节点
    dangling_nodes = np.where(M.sum(axis=1) == 0)[0]
    for node in dangling_nodes:
        M[node] = p

    #转移概率矩阵
    M = M/M.sum(axis=1)[:,None]
    for node in dangling_nodes:
        M[node] = np.repeat(0, N)
    return alpha * M + (1 - alpha) * p



def pagerank(G_Arry, alpha = 0.85):
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
    import numpy as np
    M = google_matrix(G_Arry, alpha)
    print(M)
    eigenvalues, eigenvectors = np.linalg.eig(M.T)
    # print("eigenvalues",eigenvalues)
    # print("eigenvectors",eigenvectors)
    ind = np.argmax(eigenvalues)
    # eigenvector of largest eigenvalue is at ind, normalized
    largest = np.array(eigenvectors[:, ind]).flatten().real
    # print(largest)
    norm = float(largest.sum())
    return dict(map(lambda x, y: [x, y], [x for x in range(len(G_Arry))], largest / norm))


def pagerank1(G_Arry, alpha = 0.85, max_iter=100, tol=1.0e-6):
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
    import numpy as np
    M = google_matrix(G_Arry, alpha)
    print(M)
    N = len(M)
    if N == 0:
        return M
    x = np.repeat(1.0 / N, N)
    for _ in range(max_iter):
        xlast = x
        x = np.dot(xlast, M)
        # check convergence, l1 norm
        err = sum([np.abs(x[n] - xlast[n]) for n in range(len(x))])
        if err < N * tol:
            break
    norm = x.sum()
    return x / norm


