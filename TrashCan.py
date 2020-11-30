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
        return "$v_" + str(self.index) + '$ at ' + str((self.x, self.y))

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

    def use(self):
        if self.way == 0:
            return int(self.j2)
        else:
            return int(self.j1)

    def getJuncs(self):
        return int(self.j1), int(self.j2)

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

        pos = np.array(choices([0, 1, 2], [0.1, 0.1, 0.8], k=number_of_roads))

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

        weights = self.__calcW(new_v)
        self.roads = np.array([], dtype=Road)

        for i in range(len(weights)):
            self.roads = np.append(self.roads,
                                   Road(self.juncs[new_v[i][0]], self.juncs[new_v[i][1]], weights[i], i, pos[i], i + 1))
        self.w: np.ndarray = weights.copy()

    def __calcW(self, v, with_update=False) -> np.ndarray:

        if with_update:
            # v = new weights
            weights = v.copy()
            all_close = np.array([road.isClosed() for road in self.roads], dtype=bool)
            lens = np.array([len(j1) for j1 in self.juncs])
            for i in range(len(all_close)):
                j1, j2 = self.roads[i].getJuncs()
                weights[i] += np.exp(-(lens[j1] + lens[j2]))
                if all_close[i]:
                    lens[j1] -= 1
                    lens[j2] -= 1

            for i in range(len(all_close)):
                j1, j2 = self.roads[i].getJuncs()
                weights[i] -= np.exp(-(lens[j1] + lens[j2]))

        else:
            # v = pairs of vertex, which are connected
            weights = np.array([len(self.juncs[j1]) + len(self.juncs[j2]) for (j1, j2) in v], dtype=np.float64)
            weights -= np.exp(-np.array([self.juncs[j1].airDist(self.juncs[j2]) for (j1, j2) in v], dtype=np.float64))

        return weights

    def updateWeights(self, steps_since_last_update: int = 1):
        chances = (1000 - steps_since_last_update) / 1000
        to_open_close = np.array(choices([0, 1], [chances, 1 - chances], k=len(self.roads)))

        if steps_since_last_update == 0:
            return

        for i in range(len(self.roads)):
            if to_open_close[i]:
                self.roads[i].close_open_road()
                self.w[i] = self.roads[i].w

        closed = np.array([road.isClosed() for road in self.roads])
        t_w: np.ndarray = self.w[~closed].copy()

        chances = np.random.normal(t_w.mean(), t_w.std() * steps_since_last_update ** 0.5, len(t_w))
        chances = np.abs(chances)

        self.w[~closed] = np.abs(t_w + chances / len(t_w))
        self.w = self.__calcW(self.w, True)

        for i in range(len(self.roads)):
            if not closed[i]:
                self.roads[i].update_weight(self.w[i])

            self.w[i] = self.roads[i].w

    def plot(self):
        plt.rcParams['axes.facecolor'] = 'whitesmoke'
        plt.xticks([])
        plt.yticks([])
        plt.xlim(-2, self.max_dist + 5)
        plt.ylim(-2, self.max_dist + 5)

        colors = cm.get_cmap('coolwarm')
        norm = Colors.Normalize(vmin=self.w.min(), vmax=self.w[self.w < np.inf].max())
        colors = cm.ScalarMappable(norm=norm, cmap=colors)
        cbar = plt.colorbar(colors)
        cbar.set_label('weight')
        colors = colors.to_rgba(self.w)

        names_colors = cm.get_cmap('bone')
        norm = Colors.Normalize(vmin=0, vmax=len(self.juncs))
        names_colors = cm.ScalarMappable(norm, names_colors)
        names_colors = names_colors.to_rgba([int(j1) for j1 in self.juncs])

        plt.title('The Road-Map\n$W_{ij} = deg(v_i) + deg(v_j) - e^{-dist(v_i, v_j)}$')

        px = [p[0] for p in self.all_xy]
        py = [p[1] for p in self.all_xy]

        names = [str(j1) for j1 in self.juncs]
        for i in range(len(px)):
            plt.scatter(px[i], py[i], s=len(px) * 10, label=names[i], c=names_colors[i])

        for i in range(len(self.roads)):
            px, py = self.roads[i].forPlot()
            if self.w[i] == np.inf:
                plt.arrow(px[0], py[0], dx=px[1] - px[0], dy=py[1] - py[0], ec='k', lw=2, head_width=0.3,
                          fill=True, length_includes_head=True, facecolor='k')
                continue

            plt.arrow(px[0], py[0], dx=px[1] - px[0], dy=py[1] - py[0], ec=colors[i], lw=2, head_width=0.3, fill=True,
                      length_includes_head=True, facecolor=colors[i])

        plt.legend(fancybox=True, shadow=True, facecolor='cornsilk', edgecolor='gold')
        plt.show()


# r = RoadMap(10, 17, 30)
# r.plot()

class Agent:

    def __init__(self, rMap: RoadMap, starting_junction: int, target_junction: int):

        self.Map = rMap
        self.current_junction: Junction = rMap.juncs[starting_junction]
        self.target: Junction = rMap.juncs[target_junction]

        self.x = self.current_junction.x
        self.y = self.current_junction.y
        self.possibilities = []
        self.__update_possibilities()

        self.dist_covered = 0

    def __update_possibilities(self) -> None:
        self.possibilities = []
        poss = [int(j) for j in self.current_junction.NeighborsOUT]
        roads: np.ndarray = self.Map.roads.copy()
        roads = roads.astype(Road)
        index = int(self.current_junction)

        for i in range(len(roads)):
            for j in range(len(poss)):
                if roads[i].getJuncs() == (index, poss[j]):
                    self.possibilities.append(roads[i])

        self.possibilities = np.array(self.possibilities, dtype=Road)

    def whichClosed(self) -> np.ndarray:
        self.__update_possibilities()
        return np.array([road.isClosed() for road in self.possibilities], dtype=bool)

    def getW(self) -> np.ndarray:
        # self.__update_possibilities()
        return np.array([road.w for road in self.possibilities], dtype=np.float64)

    def MoveUsing(self, using_road: int = 0) -> None:
        w = self.possibilities[using_road].w
        self.dist_covered += w
        self.current_junction: Junction = self.Map.juncs[self.possibilities[using_road].use()]
        self.Map.updateWeights(int(w / 5))
        self.__update_possibilities()
        self.x, self.y = self.current_junction.XY()

    def hasReached(self) -> bool:
        return int(self.target) == int(self.current_junction)

    def plot(self):
        plt.rcParams['axes.facecolor'] = 'whitesmoke'
        plt.xticks([])
        plt.yticks([])
        plt.xlim(-2, self.Map.max_dist + 5)
        plt.ylim(-2, self.Map.max_dist + 5)

        colors = cm.get_cmap('coolwarm')
        norm = Colors.Normalize(vmin=self.Map.w.min(), vmax=self.Map.w[self.Map.w < np.inf].max())
        colors = cm.ScalarMappable(norm=norm, cmap=colors)
        cbar = plt.colorbar(colors)
        cbar.set_label('weight')
        colors = colors.to_rgba(self.Map.w)

        names_colors = cm.get_cmap('bone')
        norm = Colors.Normalize(vmin=0, vmax=len(self.Map.juncs))
        names_colors = cm.ScalarMappable(norm, names_colors)
        names_colors = names_colors.to_rgba([int(j1) for j1 in self.Map.juncs])

        plt.title('The Road-Map\n$W_{ij} = deg(v_i) + deg(v_j) - e^{-dist(v_i, v_j)}$')

        px = [p[0] for p in self.Map.all_xy]
        py = [p[1] for p in self.Map.all_xy]

        names = [str(j1) for j1 in self.Map.juncs]
        for i in range(len(px)):
            if i == int(self.current_junction):
                plt.scatter(px[i], py[i], s=len(px) * 10, label='Agent at ' + names[i], c='tab:green')
            elif i == int(self.target):
                plt.scatter(px[i], py[i], s=len(px) * 10, label='Target at ' + names[i], c='tab:orange')
            else:
                plt.scatter(px[i], py[i], s=len(px) * 10, label=names[i], c=names_colors[i])

        for i in range(len(self.Map.roads)):
            px, py = self.Map.roads[i].forPlot()
            if self.Map.w[i] == np.inf:
                plt.arrow(px[0], py[0], dx=px[1] - px[0], dy=py[1] - py[0], ec='k', lw=2, head_width=0.3,
                          fill=True, length_includes_head=True, facecolor='k')
                continue

            plt.arrow(px[0], py[0], dx=px[1] - px[0], dy=py[1] - py[0], ec=colors[i], lw=2, head_width=0.3, fill=True,
                      length_includes_head=True, facecolor=colors[i])

        plt.legend(fancybox=True, shadow=True, facecolor='cornsilk', edgecolor='gold')
        plt.show()

# r = RoadMap(10, 17, 30)
# Ag = Agent(r, 5, 0)
# Ag.plot()
# Ag.MoveUsing(Ag.getW().argmin())
# Ag.plot()