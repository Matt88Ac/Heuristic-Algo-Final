from RoadMap import RoadMap, Junction, Road, cm, Colors
import numpy as np
import matplotlib.pyplot as plt


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
