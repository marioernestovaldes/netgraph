import numpy as np

class Graph:
    def __init__(self):
        self._edges = []
        self._vertex_count = 0

    def add_vertex(self):
        v = self._vertex_count
        self._vertex_count += 1
        return v

    def add_edge(self, v1, v2):
        self._edges.append((int(v1), int(v2)))
        return (int(v1), int(v2))

    def get_vertices(self):
        return np.arange(self._vertex_count)

    def get_edges(self):
        return np.array(self._edges)
