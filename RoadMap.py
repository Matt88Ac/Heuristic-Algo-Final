from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm
from random import choices
from matplotlib import colors as Colors


class Junction:

    def __init__(self, index: int, coordinates: tuple = (0, 0)):
        self.NeighborsIN = np.array([], dtype=Junction)
        self.NeighborsOUT: np.ndarray = self.NeighborsIN.copy()
        self.index: int = index
        self.x: float = coordinates[0]
        self.y: float = coordinates[1]

    def airDist(self, other, dtype=2) -> float:
        """
            :param other: other node. type(other) = Junction
            :param dtype: the kind of distance computed, like Euclidean, Manhattan, etc.
        """
        if dtype <= 0:
            print('dtype error')
            return 0
        return ((self.x - other.x) ** dtype + (self.y - other.y) ** dtype) ** (1 / dtype)

    def __str__(self) -> str:
        return str(self.index) + ' at ' + str((self.x, self.y))

    def __len__(self):
        return len(np.unique(self.NeighborsOUT.tolist() + self.NeighborsIN.tolist()))

    def copy(self):
        other = Junction(index=self.index, coordinates=(self.x, self.y))
        other.NeighborsIN = self.NeighborsIN.copy()
        other.NeighborsOUT = self.NeighborsOUT.copy()

        return other

    def connectIN(self, other) -> bool:
        if type(other) != Junction:
            print('connection error')
            return False

        self.NeighborsIN = np.append(self.NeighborsIN, other.copy())
        return True

    def connectOUT(self, other) -> bool:
        if type(other) != Junction:
            print('connection error')
            return False

        self.NeighborsOUT = np.append(self.NeighborsOUT, other.copy())
        return True

    def XY(self) -> tuple:
        return self.x, self.y

    def isConnectedIN(self, other) -> bool:
        return other in self.NeighborsIN

    def isConnectedOUT(self, other) -> bool:
        return other in self.NeighborsOUT

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other) -> bool:
        return int(self) == int(other)

    def __cmp__(self, other):
        if int(self) > int(other):
            return 1
        elif int(self) > int(other):
            return -1
        else:
            return 0

    def __float__(self):
        return self.index

    def __int__(self):
        return self.index

    def __lt__(self, other):
        return int(self) < int(other)

    def __gt__(self, other):
        return int(self) > int(other)


class Road:

    def __init__(self, start: Junction, end: Junction, weight, NoRoads: int, ways: int = 0, name: str = None):
        if name is None:
            self.name = np.random.randint(1, max(NoRoads, 1) * 10)
        else:
            self.name = name

        self.open = True
        self.w = weight
        self.former_w = self.w

        self.j1 = start.copy()
        self.j2 = end.copy()
        self.way = ways

        # ways = 0 <-> from start to end.
        # ways = 1 <-> from end to start.

    def update_weight(self, new_w: float):
        self.former_w = self.w
        self.w = new_w

    def close_open_road(self):
        if self.open:
            self.update_weight(np.inf)
        else:
            self.update_weight(3 * self.former_w / 4)

        self.open = not self.open

    def isClosed(self):
        return not self.open

    def forPlot(self):
        if self.way == 0:
            return [self.j1.x, self.j2.x], [self.j1.y, self.j2.y]
        else:
            return [self.j2.x, self.j1.x], [self.j2.y, self.j1.y]

    def copy(self):
        other = Road(self.j1.copy(), self.j2.copy(), self.w, 0, self.way, self.name)
        return other

    def __len__(self) -> np.float64:
        return self.w

    def __str__(self) -> str:
        return 'Road {}'.format(self.name)

    def __repr__(self) -> str:
        if self.way == 0:
            return str(self.j1) + ' -> ' + str(self.j2) + ' cost = {}'.format(self.w)
        elif self.way == 1:
            return str(self.j2) + ' -> ' + str(self.j1) + ' cost = {}'.format(self.w)
        else:
            return str(self.j2) + ' <-> ' + str(self.j1) + ' cost = {}'.format(self.w)


class RoadMap:

    def __init__(self, number_of_juncs: int, number_of_roads: int, max_dist_from_0: int = 200):
        if number_of_roads < number_of_juncs - 1:  # min edges in a connected graph
            number_of_roads = number_of_juncs - 1

        if number_of_roads > (number_of_juncs * (number_of_juncs - 1)) / 2:  # max edges in an undirected graph
            number_of_roads = number_of_juncs - 1

        self.max_dist = max_dist_from_0
        self.all_xy = [(x, y) for x, y in np.random.randint(1, max_dist_from_0 - 1, (number_of_juncs - 4, 2))]
        self.all_xy.extend([(0, 0), (0, max_dist_from_0), (max_dist_from_0, 0), (max_dist_from_0, max_dist_from_0)])
        self.juncs = np.array([Junction(index=i, coordinates=xy) for i, xy in enumerate(self.all_xy)], dtype=Junction)

        temp_vert = number_of_juncs - 2
        prufer = np.random.randint(0, temp_vert, temp_vert)

        edges = np.zeros(number_of_juncs, dtype=int)

        new_v = []

        for i in np.unique(prufer):
            edges[i] = (prufer == i).sum()

        for i in range(temp_vert):
            for j in range(number_of_juncs):
                if edges[j] == 0:
                    edges[j] = -1

                    new_v.append((j, prufer[i]))
                    edges[prufer[i]] -= 1
                    break
        j = 0
        x0 = 0
        for i in range(number_of_juncs):
            if edges[i] == 0 and j == 0:
                x0 = i
                j += 1
            elif edges[i] == 0 and j == 1:
                new_v.append((x0, i))
                break

        edges = None
        prufer = None
        temp_vert = None
        j = None
        i = None
        x0 = None

        while len(new_v) < number_of_roads:
            x = np.random.randint(0, number_of_juncs)
            y = np.random.randint(0, number_of_juncs)
            if x == y:
                continue

            if (x, y) in new_v:
                continue

            elif (y, x) in new_v:
                continue

            else:
                new_v.append((x, y))

        pos = np.array(choices([0, 1, 2], [0.125, 0.125, 0.75], k=number_of_roads))

        for i in range(number_of_roads):
            if pos[i] == 0:
                self.juncs[new_v[i][0]].connectOUT(self.juncs[new_v[i][1]])
                self.juncs[new_v[i][1]].connectIN(self.juncs[new_v[i][0]])

            elif pos[i] == 1:
                self.juncs[new_v[i][1]].connectOUT(self.juncs[new_v[i][0]])
                self.juncs[new_v[i][0]].connectIN(self.juncs[new_v[i][1]])

            else:
                pos[i] = 0
                pos = np.append(pos, 0)
                new_v.append((new_v[i][1], new_v[i][0]))
                self.juncs[new_v[i][0]].connectOUT(self.juncs[new_v[i][1]])
                self.juncs[new_v[i][1]].connectIN(self.juncs[new_v[i][0]])
                self.juncs[new_v[i][1]].connectOUT(self.juncs[new_v[i][0]])
                self.juncs[new_v[i][0]].connectIN(self.juncs[new_v[i][1]])

        weights = np.array([len(self.juncs[j1]) + len(self.juncs[j2]) for (j1, j2) in new_v], dtype=np.float64)
        weights += np.array([self.juncs[j1].airDist(self.juncs[j2]) for (j1, j2) in new_v], dtype=np.float64)

        self.roads = np.array([], dtype=Road)

        for i in range(len(weights)):
            self.roads = np.append(self.roads,
                                   Road(self.juncs[new_v[i][0]], self.juncs[new_v[i][1]], weights[i], i, pos[i], i + 1))
        self.w: np.ndarray = weights.copy()

    def updateWeights(self, steps_since_last_update: int = 1):
        closed = np.array([road.isClosed() for road in self.roads])
        t_w: np.ndarray = self.w[~closed]

        chances = np.random.normal(t_w.mean(), t_w.std() * steps_since_last_update ** 0.5, len(self.roads))
        chances = np.abs(chances)

        self.w[~closed] = np.abs(t_w + chances / len(t_w))

        chances = 1 - 1 / steps_since_last_update
        to_open_close = np.array(choices([0, 1], [chances, 1 - chances], k=len(self.roads)))

        for i in range(len(self.roads)):
            if to_open_close[i]:
                self.roads[i].close_open_road()

            elif not self.roads[i].isClosed():
                self.roads[i].update_weight(self.w[i])

            self.w[i] = self.roads[i].w

    def plot(self):
        plt.figure(facecolor='silver')
        plt.xticks([])
        plt.yticks([])
        plt.xlim(-2, self.max_dist+4)
        plt.ylim(-2, self.max_dist+4)

        colors = cm.get_cmap('jet')
        norm = Colors.Normalize(vmin=self.w.min(), vmax=self.w.max())
        colors = cm.ScalarMappable(norm=norm, cmap=colors)
        cbar = plt.colorbar(colors)
        cbar.set_label('weight')
        colors = colors.to_rgba(self.w)

        plt.title('The Road Map')

        px = [p[0] for p in self.all_xy]
        py = [p[1] for p in self.all_xy]

        names = [str(j1) for j1 in self.juncs]
        for i in range(len(px)):
            plt.scatter(px[i], py[i], s=len(px)*10, label=names[i], c=colors[i])

        for i in range(len(self.roads)):
            px, py = self.roads[i].forPlot()
            plt.arrow(px[0], py[0], dx=px[1]-px[0], dy=py[1]-py[0], ec=colors[i], lw=2, head_width=0.3, fill=True,
                      length_includes_head=True, facecolor=colors[i])

        plt.legend(fancybox=True, shadow=True, facecolor='cornsilk', edgecolor='gold')
        plt.show()


#r = RoadMap(8, 17, 30)
#r.plot()
