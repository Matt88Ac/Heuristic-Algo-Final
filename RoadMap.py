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
        self.map = folium.Map(location=start, tiles='stamenwatercolor', zoom_start=zoom)

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


rr = RoadMap((32.014191794417144, 34.773603467664515), 'hit', (32.016391794417145, 34.773603467664515), 'muir')
print(rr.xyToPixels())
