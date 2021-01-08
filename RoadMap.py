import osmnx as ox
import networkx as nx
import time
import numpy as np
from numpy import deg2rad, cos, sin, inf
import matplotlib.pyplot as plt
from typing import Union
from shapely.geometry.linestring import LineString
import matplotlib

# Global constants
EARTH_RADIUS = 6371 * 10 ** 3  # Earth radius [M]
SIGHT_RADIUS_ADDITION = 100  # The addition to the radius between the user's start and stop points [M]
NAMES_RATIO = 5  # if edge length is at least sight_radius/NAMES_RATIO it will have a name, don't put 0
IS_NAMING_ON = True
NTYPE = 'drive'  # 'drive' # 'bike'

# Global variables
sight_radius = SIGHT_RADIUS_ADDITION


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


def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)


class RoadMap:

    def __init__(self, start: Union[tuple, str], end: Union[tuple, str], network_type=NTYPE, graph_type='points'):
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

        g_nodes = self.G.nodes(data=True)

        self.start = ox.get_nearest_node(self.G, start)
        self.end = ox.get_nearest_node(self.G, end)

        print('\n-- Notice that requested coordinates are')
        print('---- src =', start, ', dst =', end)
        print('-- Actual coordinate are')
        print('---- src =', (g_nodes[self.start]['y'], g_nodes[self.start]['x']),
              ', dst =', (g_nodes[self.end]['y'], g_nodes[self.end]['x']))
        print('-- Number of nodes = ', len(g_nodes))
        print('-- Navigation type = ', NTYPE)

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

    def getName_and_Spot(self, u, v):
        data = self.G[u][v]
        try:
            name = data[0]['name'][::-1]
        except KeyError:
            name = ''

        if type(name) == list:
            name = name[0][::-1]
        try:
            spot: LineString = data[0]['geometry']
            spot = spot.centroid
        except KeyError:
            return None, None, None

        return name, spot.x, spot.y

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

        w = 0
        if current == self.end:
            prev = parents[self.nodes == self.end][0]
            path = [(prev, self.end)]
            w = self[prev, self.end]
            while prev != self.start:
                last = parents[self.nodes == prev][0]
                path.append((last, prev))
                w += self[last, prev]
                prev = last
            return path[::-1], time.time() - t, steps, w

        else:
            prev = parents[self.nodes == current][0]
            path = [(prev, current)]
            print('There is no path')
            while prev != self.start:
                last = parents[self.nodes == prev][0]
                path.append((last, prev))
                prev = last
            return path[::-1], 0, 0, w

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

        wt = 0
        if current == self.end:
            prev = parents[self.nodes == self.end][0]
            path = [(prev, self.end)]
            wt = self[prev, self.end]
            while prev != self.start:
                last = parents[self.nodes == prev][0]
                path.append((last, prev))
                wt += self[last, prev]
                prev = last

            path = path[::-1]

            if with_vis:
                t_path = []
                for edge in path:
                    t_path.append(edge)
                    if t_path[-1] != path[-1]:
                        self.plot(show=False, path=t_path)
                    else:
                        self.plot(show=True, path=t_path)
                    plt.pause(0.03)

            return path, t, steps, wt
        else:
            print('There is no path')
            return [], 0, 0

    def applyAlgorithm(self, algorithm, heuristic_function=calcGreatCircleDistanceOnEarth, with_viz=False) -> tuple:
        """
        :param with_viz: Whether or not show the process.
        :param algorithm: the wanted algorithm to apply on the graph, in order to find the shortest path from start
        to end.
        Could be one of 'A*'=0, 'Dijkstra'=1.
        :param heuristic_function: the wanted heuristic function, of type function (not str)
        """

        if algorithm == 0:
            return self.__AStar(heuristic_function, with_vis=with_viz)

        else:
            return self.__Dijkstra()

    def show_graph(self, route=None, ax=None, other_data: tuple = None):
        global sight_radius

        g_nodes = self.G.nodes(data=True)

        # build location by coordinates
        src_coordinate = ((g_nodes[self.start])['x'], (g_nodes[self.start])['y'])
        dst_coordinate = ((g_nodes[self.end])['x'], (g_nodes[self.end])['y'])
        sight_radius = int(calcGreatCircleDistanceOnEarth(src_coordinate, dst_coordinate) + SIGHT_RADIUS_ADDITION)
        fig, ax = plt.subplots(1, 1)
        paths = np.repeat('black', len(self.edges))
        lw = np.repeat(0.4, len(self.edges))
        if route:
            for v, u in route:
                cond = (self.edges[:, 0] == v) & (self.edges[:, 1] == u)
                cond += (self.edges[:, 1] == v) & (self.edges[:, 0] == u)
                paths[cond] = 'y'
                lw[cond] = 6

        # if route is not None:
        #    ox.plot_graph_route(self.G, route, route_color='y', route_linewidth=6, node_size=4, edge_linewidth=0.4,
        #                        node_color='#006666', bgcolor='white',
        #                        show=False, close=False, ax=ax)
        # else:
        ox.plot_graph(self.G, node_size=4, edge_linewidth=lw, node_color='#006666', bgcolor='white',
                      show=False, ax=ax, edge_color=paths)

        ax.scatter((g_nodes[self.start])['x'], (g_nodes[self.start])['y'], c='r', s=60, marker='.')
        ax.scatter((g_nodes[self.end])['x'], (g_nodes[self.end])['y'], c='orange', s=60, marker='*')
        if other_data:
            ax.set_title(f'time of work = {np.round(other_data[0], 4)}(s), total steps = {other_data[1]}')

        origin_high = ax.get_xlim()[1] - ax.get_xlim()[0]
        origin_width = ax.get_ylim()[1] - ax.get_ylim()[0]
        origin_area = origin_high * origin_width
        area_rel = 1
        annotations = []

        def add_annotations():
            unique_collection = set()
            for _, edge in ox.graph_to_gdfs(self.G, nodes=False).fillna('').iterrows():
                try:
                    edge['name'] = edge['name'] if type(edge['name']) == str else edge['name'][0]
                except:
                    edge['name'] = ''
                # print(edge['length'])
                if edge['length'] >= sight_radius / (NAMES_RATIO / area_rel):
                    if edge['name'] not in unique_collection:
                        unique_collection.add(edge['name'])
                        text = edge['name']
                        c = edge['geometry'].centroid
                        ann = ax.annotate(text[::-1], (c.x, c.y), c='black', size=8)
                        annotations.append(ann)

        def zoom_changed(event):
            nonlocal area_rel
            nonlocal annotations

            for ann in annotations:
                ann.remove()
            annotations.clear()

            current_high = ax.get_xlim()[1] - ax.get_xlim()[0]
            current_width = ax.get_ylim()[1] - ax.get_ylim()[0]
            area_rel = (current_high * current_width) / origin_area
            add_annotations()

        if IS_NAMING_ON:
            add_annotations()
            ax.callbacks.connect('xlim_changed', zoom_changed)
            ax.callbacks.connect('ylim_changed', zoom_changed)

        fig.set_figheight(6)
        fig.set_figwidth(9)
        move_figure(fig, 0, 0)
        plt.show()
        return fig, ax

    def plot(self, show=True, path=None, ax=None):
        paths = np.repeat('royalblue', len(self.edges))

        if ax:
            for u, v in self.edges[:, :2]:
                name, x, y = self.getName_and_Spot(u, v)
                if name is not None:
                    ax.annotate(name, (x, y), c='brown', fontsize=5)
        else:
            for u, v in self.edges[:, :2]:
                name, x, y = self.getName_and_Spot(u, v)
                if name is not None:
                    plt.annotate(name, (x, y), c='brown', fontsize=5)

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

# graph = RoadMap((32.0191, 34.7822), (32.0147, 34.7976))
# graph.plot()
# data = np.zeros((6, 3), dtype=object)
#
# metric_spaces = [calcGreatCircleDistanceOnEarth, calcEuclideanDistanceOnEarth, calcManhattanDistanceOnEarth,
#                 calcOctileDistanceOnEarth, calcChebyshevDistanceOnEarth]
## sns.set_style("dark")
# cols = ['Sphere Distance', 'Time(s)', 'Number of Steps', 'Algorithm', 'Metric Space',
#        'Total Path Weight', 'Network Type']
#
## order = ['Dijkstra', '$A^*$ - Sphere', '$A^*$ - Euclidean', '$A^*$ - Manhattan', '$A^*$ - Octile',
## '$A^*$ - Chebyshev'] for i in range(6): data[i, 2] = order[i] if i == 0: p, data[0, 0], data[0,
## 1] = graph.applyAlgorithm(1) else: p, data[i, 0], data[i, 1] = graph.applyAlgorithm(0, metric_spaces[i-1])
##
## import pandas as pd
## data = pd.DataFrame(data, columns=['Time(s)', 'NoSteps', 'Algorithm'])
## plt.subplot(1, 2, 1)
## sns.barplot(x='Algorithm', y="Time(s)", data=data, ax=plt.gca())
## plt.subplot(1, 2, 2)
## sns.barplot(x='Algorithm', y="NoSteps", data=data, ax=plt.gca())
## plt.show()
#
# data = np.zeros((60, len(cols)), dtype=object)
##
# names_metric = ['Sphere', 'Euclidean', 'Manhattan', 'Octile', 'Chebyshev']
#
##
##
# examples = [((32.0142, 34.7736), (32.0184, 34.7761)), ((32.0234, 34.7761), (32.0295, 34.7701)),
#            ((32.0136, 34.7761), (32.0194, 34.7732)), ((32.0144, 34.7711), (32.0184, 34.7767)),
#            ((32.0184, 34.7741), (32.0132, 34.7793))]
#
# spot = 0
# for i, example in enumerate(examples):
#    for net_type in ['drive', 'bike']:
#        xx, yy = example
#        dist = calcGreatCircleDistanceOnEarth(xx, yy)
#        graph = RoadMap(xx, yy, network_type=net_type)
#
#        for j in range(2):
#            if j == 0:
#                for k in range(len(names_metric)):
#                    metric = metric_spaces[k]
#                    name = names_metric[k]
#                    p, ti, st, w = graph.applyAlgorithm(0, metric)
#                    data[spot, 0] = dist
#                    data[spot, 1] = ti
#                    data[spot, 2] = st
#                    data[spot, 3] = 'A*'
#                    data[spot, 4] = name
#                    data[spot, 5] = w
#                    data[spot, 6] = net_type
#                    spot += 1
#
#            else:
#                p, ti, st, w = graph.applyAlgorithm(1, calcGreatCircleDistanceOnEarth)
#                data[spot, 0] = dist
#                data[spot, 1] = ti
#                data[spot, 2] = st
#                data[spot, 3] = 'Dijkstra'
#                data[spot, 4] = 'H(x, y)=0'
#                data[spot, 5] = w
#                data[spot, 6] = net_type
#                spot += 1
#
#        print('done')
##
# import pandas as pd
# data = pd.DataFrame(data, columns=cols)
# data.to_csv('our_data.csv')

# a = (32.0142, 34.7736)
# b = (32.0184, 34.7761)
#
# rm = RoadMap(a, b)
#
# p, ti, st = rm.applyAlgorithm(0, calcGreatCircleDistanceOnEarth)
# print(ti)
# print(st)
# rm.plot(path=p)
