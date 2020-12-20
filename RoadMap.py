import osmnx as ox
import networkx as nx
import os
from selenium import webdriver
from datetime import datetime
import numpy as np


class RoadMap:

    def __init__(self, start: tuple, end: tuple, network_type='walk'):
        dist = 900000 * ((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2) ** 0.5
        self.dist = dist
        self.st = start
        self.end = end
        self.G = ox.graph_from_point((32.0141, 34.7736), dist=dist, network_type=network_type)

    def plot(self):
        ox.plot_graph(self.G, node_color='blue', bgcolor='white', edge_color='k')


rr = RoadMap((32.0141, 34.7736), (32.0163, 34.7736))
rr.plot()
