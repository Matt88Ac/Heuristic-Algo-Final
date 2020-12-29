import osmnx as ox
import networkx as nx
import os
from datetime import datetime
import numpy as np
from numpy import deg2rad, cos, sin, inf, random
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection


def distance_between_coordinates(c1: tuple, c2: tuple) -> float:
    lat1, lon1 = c1
    lat2, lon2 = c2

    R = 6371 * 10 ** 3

    dx = deg2rad(lat2 - lat1)  # dlat
    dy = deg2rad(lon2 - lon1)  # dlon

    a = sin(dx / 2) ** 2 + cos(deg2rad(lat1)) * cos(deg2rad(lat2)) * sin(dy / 2) ** 2

    c = 2 * np.arctan2(a ** 0.5, (1 - a) ** 0.5)
    return R * c


class RoadMap:

    def __init__(self, start: tuple, end: tuple, network_type='walk'):
        dist = distance_between_coordinates(start, end) + 100
        self.dist = int(dist)
        self.G = ox.graph_from_point(start, dist=self.dist, network_type=network_type)

        self.st = ox.get_nearest_node(self.G, start)
        self.end = ox.get_nearest_node(self.G, end)

        self.st_astuple = start
        self.end_astuple = end

        self.nodes = np.array(list(self.G.nodes))
        self.edges = np.array(list(self.G.edges))

        self.pos = nx.spring_layout(self.G)

        self.algorithms = [self.__AStar, self.__Dijkstra, self.__PRM]

    def __getitem__(self, *args):
        args = args[0]
        n1, n2 = args

        def NoneCase(n3):
            cond13 = self.edges[self.edges[:, 0] == n3]
            cond23 = self.edges[self.edges[:, 1] == n3]

            if len(cond23) == 0:
                cond23 = None
            else:
                cond23 = cond23[:, 0]

            if len(cond13) == 0:
                if not cond23:
                    return []
                return cond23

            cond13 = cond13[:, 1]
            return np.unique(np.append(cond13, cond23))

        if n1 is None:
            return NoneCase(n2)

        if n2 is None:
            return NoneCase(n1)

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

    def __Dijkstra(self):
        pass

    def __AStar(self, heuristic_function=lambda dx, dy: 0):
        pass

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

        ox.plot_graph(self.G, node_color=colors, bgcolor='cornsilk', edge_color='navy',
                      edge_linewidth=3, edge_alpha=1, node_size=int(len(self.G) / 2), ax=plt.gca(), show=False)
        plt.legend(shadow=True, fancybox=True, edgecolor='gold', facecolor='wheat')
        if show:
            plt.show()


rr = RoadMap((32.0141, 34.7736), (32.0163, 34.7736))
rr.plot()
