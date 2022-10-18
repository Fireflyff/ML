
def topsort(g):
    n = len(g)
    # 获取所有入度为0的结点
    q = []
    for j in range(n):
        flag = True
        for i in range(n):
            if g[i][j] == 1:
                flag = False
                break
        if flag:
            q.insert(0, j)

    li = []  # 记录结果
    while len(q) > 0:
        # p出队，把从p出度的数据置为0
        p = q.pop()
        li.append(p)
        for i in range(n):
            if g[p][i] == 1:
                g[p][i] = 0  # 去掉连通
                # 如果结点i的入度为0则入队结点i
                in_degree_count = 0
                for u in g:
                    if u[i] == 1:
                        in_degree_count += 1
                if in_degree_count == 0:
                    q.insert(0, i)
    return li

G = [[0, 1, 1, 1, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 1, 0],
 [0, 0, 0, 0, 1, 0, 0],
 [0, 0, 0, 0, 0, 0, 1],
 [0, 0, 0, 0, 0, 0, 1],
 [0, 0, 0, 0, 0, 0, 0]]

print(topsort(G))
