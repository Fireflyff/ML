from itertools import islice
import numpy as np


class Shortest_path:
    def __init__(self, adj_Graph, source=None, target=None, weight=None, method="unweighted"):
        """
        :param adj_Graph: 图的邻接矩阵
        :param source:
        :param target:
        :param weight: 目前只支持无权图
        :param method: "unweighted", "dijkstra", "bellman-ford",目前只支持"unweighted"
        """
        # undirected graph
        self.adj_Graph = adj_Graph
        self.source = source
        self.target = target
        self.weight = weight
        self.method = method


    def shortest_path(self):
        """
        :return: 最短路径
        """
        if self.method not in ("unweighted", "dijkstra", "bellman-ford"):
            # so we don't need to check in each branch later
            raise ValueError(f"method not supported: {self.method}")
        self.method = "unweighted" if self.weight is None else self.method
        if self.source is None:
            if self.target is None:
                # Find paths between all pairs.
                if self.method == "unweighted":
                    paths = list(islice(self._all_pairs_shortest_path(self.adj_Graph), 0, self.adj_Graph.shape[0]))
                elif self.method == "dijkstra":
                    paths = []
                else:  # method == 'bellman-ford':
                    paths = []
            else:
                # Find paths from all nodes co-accessible to the target.
                if self.method == "unweighted":
                    paths = self._single_shortest_path(self.adj_Graph, self.target)
                elif self.method == "dijkstra":
                    paths = []
                else:  # method == 'bellman-ford':
                    paths = []
        else:
            if self.target is None:
                # Find paths to all nodes accessible from the source.
                if self.method == "unweighted":
                    paths = self._single_shortest_path(self.adj_Graph, self.source)
                elif self.method == "dijkstra":
                    paths = []
                else:  # method == 'bellman-ford':
                    paths = []
            else:
                # Find shortest source-target path.
                if self.method == "unweighted":
                    paths = self._bidirectional_shortest_path(self.adj_Graph, self.source, self.target)
                elif self.method == "dijkstra":
                    paths = []
                else:  # method == 'bellman-ford':
                    paths = []
        return paths

    def _single_shortest_path(self, adj_graph, node, cutoff=None):
        """
        :param adj_graph: 邻接矩阵
        :param node:
        :param cutoff:
        :return:node与所有节点的最短路径
        """
        fill_num = '1' * (len(adj_graph))
        paths = np.array([[fill_num]] * len(adj_graph))
        paths[node] = str(node)
        index = adj_graph != 0
        B = np.array([[i for i in range(len(adj_graph))]])
        if node >= len(adj_graph):
            assert node in adj_graph, f"No node {node} in graph {adj_graph}"
        if cutoff is None:
            cutoff = float("inf")
        level = 0
        firstlevel = [node]
        nextlevel = firstlevel
        while nextlevel and cutoff > level:
            thislevel = nextlevel
            nextlevel = []
            for v in thislevel:
                for w in B[0, index[v, :]]:
                    # print(type(paths[v][0]), str(w))
                    if paths[w] == fill_num:
                        path = paths[v][0] + ' ' + str(w)
                        paths[w] = path
                        nextlevel.append(w)
            level += 1
        return paths

    def _all_pairs_shortest_path(self, adj_graph, cutoff=None):
        for n in range(adj_graph.shape[0]):
            yield (self._single_shortest_path(adj_graph, n, cutoff=cutoff))

    def _bidirectional_shortest_path(self, adj_graph, source, target):
        # if source not in data or target not in data:
        #     msg = f"Either source {source} or target {target} is not in G"
        #     raise nx.NodeNotFound(msg)

        # call helper to do the real work
        results = self._bidirectional_pred_succ(adj_graph, source, target)
        pred, succ, w = results

        # build path from pred+w+succ
        path = np.array([])
        # from source to w
        while w != -1:
            path = np.append(path, w)
            w = pred[w]
        path = np.array(path[::-1])
        # from w to target
        w = succ[int(path[-1])]
        while w != -1:
            path = np.append(path, w)
            w = succ[w]

        return path.astype(int)

    def _bidirectional_pred_succ(self, adj_graph, source=0, target=5):
        if target == source:
            return (np.array([]), np.array([]), source)

        # predecesssor and successors in search
        pred = np.array([-1] * len(adj_graph[0]))
        succ = np.array([-1] * len(adj_graph[0]))

        # initialize fringes, start with forward
        forward_fringe = np.array([source])
        reverse_fringe = np.array([target])

        while len(forward_fringe) > 0 and len(reverse_fringe) > 0:

            if len(forward_fringe) <= len(reverse_fringe):
                this_level = forward_fringe
                forward_fringe = np.array([])
                for v in this_level:
                    for w in np.where(adj_graph[int(v)])[0]:
                        # print(v, w, forward_fringe, pred)
                        if w not in pred:
                            forward_fringe = np.append(forward_fringe, w)
                            if pred[w] == -1:
                                pred[w] = v
                        if w in succ:  # path found
                            return pred, succ, w
            else:
                this_level = reverse_fringe
                reverse_fringe = np.array([])
                for v in this_level:
                    for w in np.where(adj_graph[int(v)])[0]:
                        # print(v, w, reverse_fringe, succ)
                        if w not in succ:
                            if succ[w] == -1:
                                succ[w] = v
                            reverse_fringe = np.append(reverse_fringe, w)
                        if w in pred:  # found path
                            return pred, succ, w


