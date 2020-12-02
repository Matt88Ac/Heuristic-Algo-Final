import folium
import webbrowser
import os
from selenium import webdriver
from datetime import datetime
import numpy as np


class RoadMap:

    def __init__(self, start: tuple, loc_name: str, dest: tuple, dest_name: str, zoom=14.5):
        self.display_page = 'map.html'
        self.destination_tooltip = 'From'
        x0, y0 = start
        x1, y1 = dest

        self.map = folium.Map(location=start, tiles='stamenwatercolor')
        self.map.add_child(folium.LatLngPopup())

        now = datetime.now().strftime("%d-%m-%Y%H-%M-%S")
        curdir = os.getcwd().replace(os.getcwd()[2], '/') + '/Maps'
        if not os.path.exists(curdir):
            os.makedirs(curdir)
        self.dir = curdir + '/' + now
        self.fname = self.dir + '/' + self.display_page
        os.mkdir(self.dir)

        self.map.save(self.fname)
        self.driver = webdriver.Chrome()
        self.driver.get('file://' + self.fname)
        self.driver.save_screenshot(self.dir + '/out.png')

        folium.Marker(start, popup='<i>' + loc_name + '</i>', tooltip=self.destination_tooltip,
                      icon=folium.Icon(color='green')).add_to(self.map)
        folium.Marker(dest, popup='<i>' + dest_name + '</i>', tooltip='To', icon=folium.Icon(color='red')
                      ).add_to(self.map)

        self.map.save(self.fname)
        self.start = start
        self.end = dest

        self.zoom = zoom

    def get_indexes_of_square_by_coordinate(self, c):  # each square is cell in the matrix
        x = 1
        y = 1
        # calculate the picture width and high
        # take another coordinate in the range and ???
        return x, y

    def get_coordinate_by_indexes_of_square(self, x, y):  # each square is cell in the matrix
        c = 1
        return c

    def xyToPixels(self) -> tuple:
        x0, y0 = self.start
        x1, y1 = self.end

        def corToPix(x, y):
            scale = 2 ** self.zoom

            t0 = np.radians(x)
            t1 = np.radians(y)

            t0 = ((t0 + 180) / 360) * scale
            t1 = (1 - (np.log(np.tan(t1) + 1.0 / np.cos(t1)) / np.pi)) * scale / 2.0

            return [t0, t1]

        return corToPix(x0, y0), corToPix(x1, y1)

    def loadPage(self):
        webbrowser.open(self.fname)


rr = RoadMap((32.0141, 34.7736), 'hit', (32.0163, 34.7736), 'muir')
print(rr.zoom)
