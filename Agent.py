from RoadMap import RoadMap, Junction, Road
import numpy as np
import matplotlib.pyplot as plt


class Agent:

    def __init__(self, rMap: RoadMap, starting_junction: int, target_junction: int):

        self.Map = rMap
        self.current_junction: Junction = rMap.juncs[starting_junction].copy()
        self.target: Junction = rMap.juncs[target_junction].copy()

        self.x = self.current_junction.x
        self.y = self.current_junction.y
        self.possibilities = []
        self.__update_possibilities()

    def __update_possibilities(self):
        self.possibilities = []
        poss = [int(j) for j in self.current_junction.NeighborsOUT]
        roads: np.ndarray = self.Map.roads.copy()
        roads = roads.astype(Road)
        index = int(self.current_junction)

        for i in range(len(roads)):
            for j in range(len(poss)):
                if roads[i].getJuncs() == (index, poss[j]):
                    self.possibilities.append(roads[i])
