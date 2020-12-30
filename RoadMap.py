import osmnx as ox
import networkx as nx
import os
from datetime import datetime
import numpy as np
from numpy import deg2rad, cos, sin, inf, random
import matplotlib.pyplot as plt
import heapq


def CoordinatesEuclidean(c1: tuple, c2: tuple) -> float:
    lat1, lon1 = c1
    lat2, lon2 = c2

    R = 6371 * 10 ** 3

    dx = deg2rad(abs(lat2 - lat1))  # dlat
    dy = deg2rad(abs(lon2 - lon1))  # dlon

    a = sin(dx / 2) ** 2 + cos(deg2rad(lat1)) * cos(deg2rad(lat2)) * sin(dy / 2) ** 2

    c = 2 * np.arctan2(a ** 0.5, (1 - a) ** 0.5)
    return R * c


def CoordinatesManhattan(c1: tuple, c2: tuple):
    lat1, lon1 = c1
    lat2, lon2 = c2
    R = 6371 * 10 ** 3

    def getR(x0, x1):
        a = sin(deg2rad(abs(x0 - x1)) / 2) ** 2
        c = 2 * np.arctan2(a ** 0.5, (1 - a) ** 0.5)
        return R * c

    return getR(lat1, lat2) + getR(lon1, lon2)


def CoordinatesChebyshev(c1: tuple, c2: tuple):
    lat1, lon1 = c1
    lat2, lon2 = c2
    R = 6371 * 10 ** 3

    def getR(x0, x1):
        a = sin(deg2rad(abs(x0 - x1)) / 2) ** 2
        c = 2 * np.arctan2(a ** 0.5, (1 - a) ** 0.5)
        return R * c

    return max(getR(lat1, lat2), getR(lon1, lon2))


class RoadMap:

    def __init__(self, start: tuple, end: tuple, network_type='walk'):
        dist = CoordinatesEuclidean(start, end) + 100
        self.dist = int(dist)
        self.G = ox.graph_from_point(start, dist=self.dist, network_type=network_type)
        data = self.G.nodes(data=True)

        self.st = ox.get_nearest_node(self.G, start)
        self.end = ox.get_nearest_node(self.G, end)

        self.nodes = np.array(list(self.G.nodes))
        self.edges = np.array(list(self.G.edges))

        self.coordinates = []
        for node in data:
            self.coordinates.append((node[1]['y'], node[1]['x']))

        self.coordinates = np.array(self.coordinates)
        self.coordinates = self.coordinates[self.nodes.argsort()]
        self.nodes.sort()
        self.pos = nx.spring_layout(self.G)

        self.algorithms = [self.__AStar, self.__Dijkstra, self.__PRM]

    def __getitem__(self, *args):
        args = args[0]
        n1, n2 = args

        def NoneCase(n3):  # returns each of n3's neighbors
            return np.unique(nx.all_neighbors(self.G, n3))

        if n1 is None:
            return NoneCase(n2)

        if n2 is None:
            return NoneCase(n1)
        # returns the edge consists of (n1, n2) if exists
        cond11 = self.edges[:, 0] == n1
        cond12 = self.edges[:, 1] == n2

        cond1 = cond11 & cond12

        cond11 = self.edges[:, 1] == n1
        cond12 = self.edges[:, 0] == n2

        cond2 = cond11 & cond12

        if cond1.sum() > 0:
            return self.G[n1][n2]
        elif cond2.sum() > 0:
            return self.G[n2][n1]

        return None

    def __len__(self):  # return the number of vertices in the graph G
        return len(self.G)

    def fromOsPoint_to_tuple(self, p) -> tuple:  # translates the node's name to a coordinate
        return self.coordinates[self.nodes == p][0][0], self.coordinates[self.nodes == p][0][1]

    def __Dijkstra(self):
        pass

    def __AStar(self, heuristic_function=CoordinatesEuclidean, g=None, f=None):
        if not f:
            f = np.zeros(len(self))
        if not g:
            g: np.ndarray = np.ones(len(self))
        h: np.ndarray = f.copy()

        path = [self.st]

        closed = []
        opened = [self.st]

        end = self.end
        start = self.st

        end_tup = self.fromOsPoint_to_tuple(end)

        current = None

        g[self.nodes == start] = 0

        while len(opened) > 0 and current is not end:
            current = heapq.heappop(opened)
            neighbors = self[current]

            for ne in neighbors:
                h = heuristic_function(self.fromOsPoint_to_tuple(ne), end_tup)





    def __PRM(self):
        pass

    def applyAlgorithm(self, algorithm, heuristic_function=lambda dx, dy: 0):
        """
        :param algorithm: the wanted algorithm to apply on the graph, in order to find the shortest path from start
        to end.
        Could be one of 'A*'=0, 'Dijkstra'=1, 'PRM'=2.
        :param heuristic_function: the wanted heuristic function, of type function (not str)
        """

        if algorithm != 0:
            heuristic_function = lambda dx, dy: 0

        else:
            self.algorithms[0](heuristic_function)

    def plot(self, show=True):
        colors = np.repeat('dimgray', len(self.G))
        colors[self.nodes == self.st] = 'lime'
        colors[self.nodes == self.end] = 'r'
        plt.plot([0], [0], label='start', c='lime')
        plt.plot([0], [0], label='goal', c='r')
        plt.plot([0], [0], label='nodes', c='dimgray')
        s = int(8 * len(self.G) / 13)
        ox.plot_graph(self.G, node_color=colors, bgcolor='cornsilk', edge_color='navy',
                      edge_linewidth=3, edge_alpha=1, node_size=s, ax=plt.gca(), show=False)
        plt.legend(shadow=True, fancybox=True, edgecolor='gold', facecolor='wheat')
        if show:
            plt.show()


rr = RoadMap((32.0141, 34.7736), (32.0183, 34.7761))
rr.plot()
