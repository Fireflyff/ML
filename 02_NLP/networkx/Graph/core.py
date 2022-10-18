import numpy as np


class Core:
    def __init__(self, adj_graph):
        self.adj_graph = adj_graph

    def k_core(self, k):
        #old_temp存储每个node的degree
        old_temp = np.where(sum(self.adj_graph) >= k)
        #res存储最终的子图结构
        res = np.array([[0] * len(self.adj_graph)] * len(self.adj_graph))

        def subgraph():
            nonlocal res
            res = np.array([[0] * len(self.adj_graph)] * len(self.adj_graph))
            for i in old_temp[0]:
                if sum(self.adj_graph[i][old_temp]) >= k:
                    res[i][old_temp] = self.adj_graph[i][old_temp]

        subgraph()
        new_temp = np.where(sum(res) >= k)
        while len(new_temp[0]) != len(old_temp[0]):
            if len(new_temp[0]) == 0:
                res = np.array([[0] * len(self.adj_graph)] * len(self.adj_graph))
                break
            old_temp = new_temp
            subgraph()
            new_temp = np.where(sum(res) >= k)
        return res

    # def k_shell(self):