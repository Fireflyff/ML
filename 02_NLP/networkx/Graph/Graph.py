import numpy as np
import copy


class Graph:
    def __init__(self, edges=None, nodes=None):
        self.edges = []
        self.nodes = []
        if edges:
            for edge in edges:
                self.add_edge(edge[0], edge[1])
            self.add_nodes_from([edge[0], edge[1]])
        if nodes:
            self.add_nodes_from(nodes)

    def has_edge(self, source, target):
        node_name = (source, target)
        return node_name in self.edges

    def has_node(self, node):
        return node in self.nodes

    def add_node(self, node):
        if node not in self.nodes:
            self.nodes.append(node)

    def add_edge(self, source, target):
        if ((source, target) in self.edges) or ((target, source) in self.edges):
            return
        if source not in self.nodes:
            self.nodes.append(source)
        if target not in self.nodes:
            self.nodes.append(target)
        self.edges.append((source, target))

    def add_nodes_from(self, nodes):
        for n in nodes:
            if n not in self.nodes:
                self.nodes += [n]

    def add_edges_from(self, edges=None):
        if edges is None:
            return None
        for edge in edges:
            self.add_edge(edge[0], edge[1])

    def get_edges(self):
        return self.edges

    def get_nodes(self):
        return self.nodes

    def remove_node(self, node):
        """
        Remove the node and all adjacent edges
        Attempting to remove a non-existent node will assert sth.

        :param node:
            A node in the graph

        Assert
        ----------
        If node is not in the graph.

        """
        if node not in self.nodes:
            assert node in self.nodes, f"The node {node} is not in graph"
        else:
            # remove node
            self.nodes.remove(node)
            # remove all adjacent edges
            edge_list = copy.copy(self.edges)
            for e in edge_list:
                if node in e:
                    self.remove_edge(e[0], e[1])

    def remove_edge(self, source, target):
        if not (self.has_edge(source, target) or self.has_edge(target, source)):
            assert (source, target) in self.edges, f"The edge {source}-{target} is not in the graph"
        else:
            if self.has_edge(source, target):
                self.edges.remove((source, target))
            if self.has_edge(target, source):
                self.edges.remove((target, source))

    def remove_nodes_from(self, nodes):
        for n in nodes:
            self.remove_node(n)

    def remove_edges_from(self, edges):
        if edges is None:
            return None
        for edge in edges:
            self.remove_edge(edge[0], edge[1])

    def to_adj_matrix(self, nodelist=None):
        if nodelist is None:
            nodelist = self.nodes
        keys = [k for k in range(len(self.nodes))]
        reflection = dict(zip(nodelist, keys))
        adj = np.zeros((len(self.nodes), len(self.nodes)), dtype=int)
        for edge in self.edges:
            i, j = reflection[edge.source], reflection[edge.target]
            adj[i, j] += 1
            if i != j:
                adj[j, i] += 1
        return adj

    def to_numpy_array(self, nodelist=None):
        if nodelist is None:
            nodelist = self.nodes
        keys = [k for k in range(len(self.nodes))]
        reflection = dict(zip(nodelist, keys))
        adj = np.zeros((len(self.nodes), len(self.nodes)), dtype=int)
        for edge in self.edges:
            i, j = reflection[edge.source], reflection[edge.target]
            adj[i, j] = 1
            adj[j, i] = 1
        return adj
