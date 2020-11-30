import folium
import webbrowser
import os
from selenium import webdriver
import html


class RoadMap:

    def __init__(self):
        self.display_page = 'map.html'
        self.destination_tooltip = 'Destination'
        self.map = folium.Map

        f = os.getcwd().replace(os.getcwd()[2], '/') + '/map.html'

        driver = webdriver.Chrome()
        driver.get('file://' + f)
        driver.save_screenshot('out.png')


HIT_COORDINATES = (32.014191794417144, 34.773603467664515)
DISPLAY_PAGE = 'map.html'
DESTINATION_TOOLTIP = 'Destination'
foliMap = folium.Map(location=HIT_COORDINATES, tiles='Stamen Toner', zoom_start=15.5, png_enabled=True)
tooltip = DESTINATION_TOOLTIP
folium.Marker(HIT_COORDINATES, popup='<i>HIT</i>', tooltip=tooltip).add_to(foliMap)
foliMap.add_child(folium.LatLngPopup())
folium.Marker(
    [46.8354, -121.7325],
    popup='Camp Muir'
).add_to(foliMap)
foliMap.add_child(folium.CircleMarker(popup='Waypoint', location=HIT_COORDINATES))
foliMap.save(DISPLAY_PAGE)
