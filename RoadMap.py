import osmnx as ox
import networkx as nx
import time
import numpy as np
from numpy import deg2rad, cos, sin, inf
import matplotlib.pyplot as plt
from typing import Union

EARTH_RADIUS = 6371 * 10 ** 3  # Earth radius [M]
SIGHT_RADIUS_ADDITION = 100  # The addition to the radius between the user's start and stop points [M]


# Calculates the distance between two coordinates on earth (a sphere)
def calcGreatCircleDistanceOnEarth(c1: tuple, c2: tuple) -> float:
    lat1, lon1 = c1
    lat2, lon2 = c2

    # convert angles from degree to radian
    dx = deg2rad(abs(lat2 - lat1))  # dlat
    dy = deg2rad(abs(lon2 - lon1))  # dlon

    a = sin(dx / 2) ** 2 + cos(deg2rad(lat1)) * cos(deg2rad(lat2)) * sin(dy / 2) ** 2
    c = 2 * np.arctan2(a ** 0.5, (1 - a) ** 0.5)
    return EARTH_RADIUS * c


def calcManhattanDistanceOnEarth(c1: tuple, c2: tuple):
    lat1, lon1 = c1
    lat2, lon2 = c2

    def getR(x0, x1):
        a = sin(deg2rad(abs(x0 - x1)) / 2) ** 2
        c = 2 * np.arctan2(a ** 0.5, (1 - a) ** 0.5)
        return EARTH_RADIUS * c

    return getR(lat1, lat2) + getR(lon1, lon2)


def calcEuclideanDistanceOnEarth(c1: tuple, c2: tuple):
    lat1, lon1 = c1
    lat2, lon2 = c2

    def getR(x0, x1):
        a = sin(deg2rad(abs(x0 - x1)) / 2) ** 2
        c = 2 * np.arctan2(a ** 0.5, (1 - a) ** 0.5)
        return EARTH_RADIUS * c

    return np.sqrt(getR(lat1, lat2) ** 2 + getR(lon1, lon2) ** 2)


def calcChebyshevDistanceOnEarth(c1: tuple, c2: tuple):
    lat1, lon1 = c1
    lat2, lon2 = c2

    def getR(x0, x1):
        a = sin(deg2rad(abs(x0 - x1)) / 2) ** 2
        c = 2 * np.arctan2(a ** 0.5, (1 - a) ** 0.5)
        return EARTH_RADIUS * c

    return max(getR(lat1, lat2), getR(lon1, lon2))


def calcOctileDistanceOnEarth(c1: tuple, c2: tuple):
    lat1, lon1 = c1
    lat2, lon2 = c2

    def getR(x0, x1):
        a = sin(deg2rad(abs(x0 - x1)) / 2) ** 2
        c = 2 * np.arctan2(a ** 0.5, (1 - a) ** 0.5)
        return EARTH_RADIUS * c

    mx = max(getR(lat1, lat2), getR(lon1, lon2))
    mn = min(getR(lat1, lat2), getR(lon1, lon2))

    return (np.sqrt(2) - 1) * mn + mx


class RoadMap:

    def __init__(self, start: Union[tuple, str], end: Union[tuple, str], network_type='drive', graph_type='points'):
        if graph_type == 'points':
            self.dist = int(calcGreatCircleDistanceOnEarth(start, end) + SIGHT_RADIUS_ADDITION)
            self.G = ox.graph_from_point(start, dist=self.dist, network_type=network_type)

        else:
            g1, start = ox.graph_from_address(start, 20, return_coords=True)
            g2, end = ox.graph_from_address(end, 20, return_coords=True)
            self.dist = int(calcGreatCircleDistanceOnEarth(start, end) + SIGHT_RADIUS_ADDITION)
            self.G = ox.graph_from_point(start, dist=self.dist, network_type=network_type)

        self.G = ox.add_edge_speeds(self.G, hwy_speeds={'motorway': 130, 'trunk': 110, 'primary': 70,
                                                        'secondary': 50, 'tertiary': 50, 'unclassified': 30,
                                                        'residential': 30, 'steps': 0, 'trunk_link': 70,
                                                        'motorway_link': 70, 'primary_link': 40,
                                                        'secondary_link': 20, 'tertiary_link': 20, 'service': 10,
                                                        'living_street': 20}, fallback=1)
        self.G = ox.add_edge_travel_times(self.G)

        self.start = ox.get_nearest_node(self.G, start)
        self.end = ox.get_nearest_node(self.G, end)

        self.nodes = np.array(list(self.G.nodes))
        self.edges = np.array(list(self.G.edges), dtype=float)

        w = np.ones(self.edges.shape[0], dtype=float) * inf
        for u, v, d in self.G.edges(data=True):
            w[(self.edges[:, 0] == u) & (self.edges[:, 1] == v)] = d['travel_time']

        self.edges[:, 2] = w.copy()

        data = self.G.nodes(data=True)
        self.coordinates = []
        for node in data:
            self.coordinates.append((node[1]['y'], node[1]['x']))

        self.coordinates = np.array(self.coordinates)
        self.coordinates = self.coordinates[self.nodes.argsort()]
        self.nodes.sort()
        self.pos = nx.spring_layout(self.G)

        self.blocked = []

    def __getitem__(self, *args):
        args = args[0]
        n1, n2 = args

        def NoneCase(n3):  # returns each of n3's neighbors
            return np.array(list(nx.neighbors(self.G, n3)))

        if n1 is None:
            return NoneCase(n2)

        if n2 is None:
            return NoneCase(n1)
        # returns the weight of the edge consists of (n1, n2) if exists
        cond11 = self.edges[:, 0] == n1
        cond12 = self.edges[:, 1] == n2

        cond1 = cond11 & cond12

        if cond1.sum() > 0:
            return self.edges[cond1][:, 2][0]
        # elif cond2.sum() > 0:
        #    return self.edges[cond2][:, 2][0]

        return inf

    # Returns the number of vertices in the graph G
    def __len__(self):
        return len(self.G)

    def blockRoad(self, u, v):
        cond = self.edges[:, 0] == u
        cond &= self.edges[:, 1] == v

        if cond.sum() == 0:
            return False

        self.blocked.append((u, v, self.edges[cond][:, 2][0]))
        self.edges[cond] = np.array([u, v, inf])
        return True

    def openRoad(self, u, v):
        cond = self.edges[:, 0] == u
        cond &= self.edges[:, 1] == v

        if cond.sum() == 0 or len(self.blocked) == 0:
            return False

        for uu, vv, w in self.blocked:
            if uu == u and vv == v:
                self.edges[cond] = np.array([u, v, w])
                self.blocked.remove((uu, vv, w))
                return True

        return False

    def fromOsPoint_to_tuple(self, pt) -> tuple:  # translates the node's name to a coordinate
        return self.coordinates[self.nodes == pt][0][0], self.coordinates[self.nodes == pt][0][1]

    def __Dijkstra(self):
        opened = np.array([self.start])
        ww = np.array([0])
        closed = []
        steps = 0
        distances = np.ones(len(self.nodes)) * inf
        current = self.start

        distances[self.nodes == self.start] = 0

        t = time.time()
        parents = np.zeros_like(self.nodes)

        while len(opened) > 0 and current != self.end:
            current = opened[0]
            ww = ww[opened != current]
            opened = opened[opened != current]
            closed.append(current)
            neighbors = self[current, None]
            steps += 1
            curr_dist = distances[self.nodes == current][0]

            for neighbor in neighbors:
                w = self[current, neighbor]
                if current == self.start:
                    distances[self.nodes == neighbor] = w + curr_dist
                    parents[self.nodes == neighbor] = current
                else:
                    if w + curr_dist < distances[self.nodes == neighbor][0]:
                        distances[self.nodes == neighbor] = w + curr_dist
                        parents[self.nodes == neighbor] = current
                        if neighbor in opened:
                            ww[opened == neighbor] = w + curr_dist

                if neighbor not in closed:
                    opened = np.append(opened, neighbor)
                    ww = np.append(ww, distances[self.nodes == neighbor][0])
                    opened = opened[ww.argsort()]
                    ww.sort()

        if current == self.end:
            prev = parents[self.nodes == self.end][0]
            path = [(prev, self.end)]
            while prev != self.start:
                last = parents[self.nodes == prev][0]
                path.append((last, prev))
                prev = last
            return path[::-1], time.time() - t, steps

        else:
            prev = parents[self.nodes == current][0]
            path = [(prev, current)]
            print('There is no path')
            while prev != self.start:
                last = parents[self.nodes == prev][0]
                path.append((last, prev))
                prev = last
            return path[::-1], 0, 0

    def __AStar(self, heuristic_function=calcGreatCircleDistanceOnEarth, with_vis=False):
        f = np.ones(len(self)) * inf
        g: np.ndarray = np.zeros(len(self))
        h: np.ndarray = g.copy()

        parents = np.zeros_like(self.nodes)

        h[self.nodes == self.start] = heuristic_function(self.fromOsPoint_to_tuple(self.start),
                                                         self.fromOsPoint_to_tuple(self.end))
        f[self.nodes == self.start] = h[self.nodes == self.start][0]

        closed = []
        opened = np.array([self.start])
        wt = np.array([0])

        steps = 0
        current = None
        t = time.time()

        if with_vis:
            plt.figure()
            plt.ion()

        while len(opened) > 0 and current is not self.end:
            current = opened[0]
            if current == self.end:
                t = time.time() - t
                break
            wt = wt[opened != current]
            opened = opened[opened != current]
            closed.append(current)

            steps += 1
            neighbors = self[current, None]

            for ne in neighbors:
                h1 = heuristic_function(self.fromOsPoint_to_tuple(ne),
                                        self.fromOsPoint_to_tuple(self.end))

                g[self.nodes == ne] = g1 = self[current, ne] + g[self.nodes == current][0]
                if ne in closed and f[self.nodes == ne][0] > h1 + g1:
                    closed.remove(ne)
                elif ne in closed and f[self.nodes == ne][0] <= h1 + g1:
                    continue

                parents[self.nodes == ne] = current
                h[self.nodes == ne] = h1
                f[self.nodes == ne] = h1 + g1
                wt = np.append(wt, h1 + g1)
                opened = np.append(opened, ne)

            opened = opened[wt.argsort()]
            wt.sort()

        # minF = np.min(f[cond])
        # candidate = self.nodes[cond][f[cond] == minF][0]
        #
        # if (self.end in neighbors) and (f[self.nodes == self.end][0] == minF):
        #     path.append((current, self.end))
        #     current = self.end
        #     if with_vis:
        #         self.plot(show=True, path=path)
        #     continue
        #
        # path.append((current, candidate))
        # if with_vis:
        #     self.plot(show=False, path=path)
        #     plt.pause(0.01)
        if current == self.end:
            prev = parents[self.nodes == self.end][0]
            path = [(prev, self.end)]
            while prev != self.start:
                last = parents[self.nodes == prev][0]
                path.append((last, prev))
                prev = last
            return path[::-1], t, steps
        else:
            print('There is no path')
            return [], 0, 0

    def applyAlgorithm(self, algorithm, heuristic_function=calcGreatCircleDistanceOnEarth, with_viz=False) -> tuple:
        """
        :param algorithm: the wanted algorithm to apply on the graph, in order to find the shortest path from start
        to end.
        Could be one of 'A*'=0, 'Dijkstra'=1.
        :param heuristic_function: the wanted heuristic function, of type function (not str)
        """

        if algorithm == 0:
            return self.__AStar(heuristic_function, with_vis=with_viz)

        else:
            return self.__Dijkstra()

    def plot(self, show=True, path=None, ax=None):
        paths = np.repeat('royalblue', len(self.edges))

        if path is not None:
            if len(path) > 0 and show:
                if ax:
                    plt.plot([0], [0], label='path', c='gold')

                else:
                    plt.plot([0], [0], label='path', c='gold')

            if not show and ax:
                ax.plot([0], [0], label='path', c='gold')
            for v, u in path:
                cond = (self.edges[:, 0] == v) & (self.edges[:, 1] == u)
                cond += (self.edges[:, 1] == v) & (self.edges[:, 0] == u)
                paths[cond] = 'gold'

        if len(self.blocked) > 0:
            for u, v, w in self.blocked:
                cond = (self.edges[:, 0] == v) & (self.edges[:, 1] == u)
                cond += (self.edges[:, 1] == v) & (self.edges[:, 0] == u)
                paths[cond] = 'darkred'

        colors = np.repeat('black', len(self.G))
        colors[self.nodes == self.start] = 'lime'
        colors[self.nodes == self.end] = 'r'

        if not ax:
            ox.plot_graph(self.G, node_color=colors, edge_color=paths,
                          edge_linewidth=3, edge_alpha=1, ax=plt.gca(), show=False)
        else:
            ox.plot_graph(self.G, node_color=colors, edge_color=paths,
                          edge_linewidth=3, edge_alpha=1, ax=ax, show=False)
        if not ax:
            plt.plot([0], [0], label='start', c='lime')
            plt.plot([0], [0], label='goal', c='r')
            plt.plot([0], [0], label='nodes', c='black')
            plt.legend(shadow=True, fancybox=True, edgecolor='gold', facecolor='wheat')
        else:
            ax.plot([0], [0], label='start', c='lime')
            ax.plot([0], [0], label='goal', c='r')
            ax.plot([0], [0], label='nodes', c='black')
            ax.legend(shadow=True, fancybox=True, edgecolor='gold', facecolor='wheat')

        if show:
            plt.ioff()
            plt.show()


rm = RoadMap((32.0142, 34.7736), (32.0184, 34.7761))
p, ti, st = rm.applyAlgorithm(0, calcGreatCircleDistanceOnEarth)
print(ti)
print(st)
rm.plot(path=p)
