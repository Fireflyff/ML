import numpy as np
class FastNewman:
    def __init__(self, adj_Graph):
        self.A = adj_Graph
        self.num_node = len(self.A)
        self.num_edge = sum(sum(self.A))  # 边数
        self.c = {}  # 记录所有Q值对应的社团分布

    def merge_community(self, iter_num, detaQ, e, b):
        (I, J) = np.where(detaQ == np.amax(detaQ))
        for m in range(len(I)):
            e[J[m], :] = e[I[m], :] + e[J[m], :]
            e[I[m], :] = 0
            e[:, J[m]] = e[:, I[m]] + e[:, J[m]]
            e[:, I[m]] = 0
            b[J[m]] = b[J[m]] + b[I[m]]

        e = np.delete(e, I, axis=0)
        e = np.delete(e, I, axis=1)
        I = sorted(list(set(I)), reverse=True)
        for i in I:
            b.remove(b[i])  # 删除第I组社团，（都合并到J组中了）
        self.c[iter_num] = b.copy()
        return e, b

    def Run_FN(self):
        e = self.A / self.num_edge  # 社区i,j连边数量占总的边的比例
        a = np.sum(e, axis=0)  # e的列和，表示与社区i中节点相连的边占总边数的比例
        b = [[i] for i in range(self.num_node)]  # 本轮迭代的社团分布
        Q = []
        iter_num = 0
        while len(e) > 1:
            num_com = len(e)
            detaQ = -np.power(10, 9) * np.ones((self.num_node, self.num_node))  # detaQ可能为负数，初始设为负无穷
            for i in range(num_com - 1):
                for j in range(i + 1, num_com):
                    if e[i, j] != 0:
                        detaQ[i, j] = 2 * (e[i, j] - a[i] * a[j])
            if np.sum(detaQ + np.power(10, 9)) == 0:
                break

            e, b = self.merge_community(iter_num, detaQ, e, b)

            a = np.sum(e, axis=0)
            # 计算Q值
            Qt = 0.0
            for n in range(len(e)):
                Qt += e[n, n] - a[n] * a[n]
            Q.append(Qt)
            iter_num += 1
        max_Q, community = self.get_community(Q)
        return max_Q, community

    def get_community(self, Q):
        max_k = np.argmax(Q)
        community = self.c[max_k]
        return Q[max_k], community

