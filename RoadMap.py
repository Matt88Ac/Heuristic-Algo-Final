import folium
import webbrowser
import os
from selenium import webdriver
from datetime import datetime


class RoadMap:

    def __init__(self, start: tuple, loc_name: str, dest: tuple, dest_name: str, zoom=14.5):
        self.display_page = 'map.html'
        self.destination_tooltip = 'From'
        self.map = folium.Map(location=start, tiles='stamenwatercolor', zoom_start=zoom)

        now = datetime.now().strftime("%d-%m-%Y%H-%M-%S")
        curdir = os.getcwd().replace(os.getcwd()[2], '/') + '/Maps'
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


rr = RoadMap((32.014191794417144, 34.773603467664515), 'hit', (32.016391794417145, 34.773603467664515), 'muir')
