import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from RoadMap import RoadMap, calcChebyshevDistanceOnEarth, calcEuclideanDistanceOnEarth, calcGreatCircleDistanceOnEarth
from RoadMap import calcManhattanDistanceOnEarth, calcOctileDistanceOnEarth
from matplotlib.figure import Figure
import matplotlib as mpl
from matplotlib import style

mpl.use("TkAgg")
style.use('ggplot')

root = tk.Tk()
root.iconbitmap(default='icon.ico')


def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()


root.protocol("WM_DELETE_WINDOW", on_closing)


class OurApp(tk.Frame):

    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.winfo_rgb('MediumPurple1')
        self.winfo_toplevel().title("Final Project in Heuristic and Approximation Algorithms")
        self.path = None
        self.button1 = None
        self.button2 = None
        # self.button3 = None
        self.label_time = None
        self.label_error = tk.Label(self, text='No graph to show', width=15)
        self.label_error.config(font=('Arial', 16))

        self.canvas1 = tk.Canvas(self, width=800, height=400)
        self.canvas1.pack()
        self.getter = 'points'

        self.button4 = None
        self.button5 = None
        self.button6 = None
        self.fig = None
        self.ax = None

        self.defAlgorithms()

        self.__ApplySettings()
        self.create_widgets()
        self.graph = RoadMap

    def create_widgets(self):
        self.button1 = tk.Button(self, text='  Set Graph  ', command=self.__setWorld, bg='lightsteelblue2',
                                 font=('Arial', 11, 'bold'), width=15)
        self.canvas1.create_window(100, 120, window=self.button1)

        self.button2 = tk.Button(self, text='Apply Algorithm', command=self.applyAlgorithm, bg='lightsteelblue2',
                                 font=('Arial', 11, 'bold'), width=15)
        self.canvas1.create_window(100, 170, window=self.button2)

        # self.button3 = tk.Button(self, text='  Show Graph  ', command=self.show, bg='lightsteelblue2',
        #                          font=('Arial', 11, 'bold'), width=15)
        # self.canvas1.create_window(100, 220, window=self.button3)

        self.button4 = tk.Button(self, text='  Clear Graph  ', command=self.__clearWorld, bg='lightsteelblue2',
                                 font=('Arial', 11, 'bold'), width=15)
        self.canvas1.create_window(100, 220, window=self.button4)

        self.button5 = tk.Button(self, text='Exit Application', command=self.master.destroy, bg='lightsteelblue2',
                                 font=('Arial', 11, 'bold'), width=15)
        self.canvas1.create_window(100, 270, window=self.button5)

    def defAlgorithms(self):
        self.type = tk.StringVar()
        choices = ('points', 'address')
        self.type.set(self.getter)
        self.typer = tk.OptionMenu(self, self.type, *choices)
        self.typer.pack()
        self.canvas1.create_window(750, 150, window=self.typer)

        label = tk.Label(self, text='Input type:', width=15)
        label.config(font=('Arial', 9))
        self.canvas1.create_window(740, 120, window=label)

        label = tk.Label(self, text='Metric Space:', width=15)
        label.config(font=('Arial', 9))
        self.canvas1.create_window(640, 240, window=label)

        label = tk.Label(self, text='Algorithm:', width=15)
        label.config(font=('Arial', 9))
        self.canvas1.create_window(740, 240, window=label)

        self.choiceDist = tk.StringVar()
        choices = ("Sphere", "Euclidean", "Manhattan", "Chebyshev", "Octile")
        self.choiceDist.set(choices[0])
        self.distances = tk.OptionMenu(self, self.choiceDist, *choices)
        self.distances.pack()
        self.canvas1.create_window(640, 270, window=self.distances)

        self.choiceAlgo = tk.StringVar()
        choices = ('A*', 'Dijkstra')
        self.choiceAlgo.set(choices[0])
        self.algorithms = tk.OptionMenu(self, self.choiceAlgo, *choices)
        self.algorithms.pack()
        self.canvas1.create_window(740, 270, window=self.algorithms)

    def __setWorld(self):
        start = self.getStart()
        end = self.getEnd()
        if start[0] is None or start[1] is None or end[0] is None or end[1] is None:
            raise ValueError
        if self.getter != 'points':
            self.graph = RoadMap(start, end, graph_type='address')
        else:
            self.graph = RoadMap(start, end)
        self.show()

    def __clearWorld(self):
        if self.getter != self.type.get():
            self.getter = self.type.get()
            self.canvas1.delete('all')
            self.defAlgorithms()
            self.create_widgets()
            self.__ApplySettings()

        plt.close('all')
        self.fig = None
        self.ax = None
        self.path = None

    def show(self):
        other_data = None
        if self.path is not None:
            other_data = (self.time, self.steps)
        try:
            plt.close('all')
            self.graph.show_graph(route=self.path, other_data=other_data)
        except TypeError:
            pass

    def __ApplySettings(self):
        label = tk.Label(self, text='Shortest Path Finder On Earth')
        label.config(font=('Arial', 20))
        self.canvas1.create_window(400, 50, window=label)
        if self.type.get() == 'points':
            self.latlon1 = tk.StringVar(self)
            self.latlon0 = tk.StringVar(self)

            entery = tk.Entry(self, textvariable=self.latlon0)
            self.canvas1.create_window(400, 180, window=entery, width=250)

            entery = tk.Entry(self, textvariable=self.latlon1)
            self.canvas1.create_window(400, 230, window=entery, width=250)

            label = tk.Label(self, text='Start Coordinate: lat,lon')
            label.config(font=('Arial', 9))
            self.canvas1.create_window(400, 160, window=label)

            label = tk.Label(self, text='Goal Coordinate: lat,lon')
            label.config(font=('Arial', 9))
            self.canvas1.create_window(400, 210, window=label)

            # self.label_time = tk.Label(self, text=f'time of work = ?(s), total steps = ?')
            # self.label_time.config(font=('Arial', 9))
            # self.canvas1.create_window(400, 280, window=self.label_time)
            #
            # self.label_error = tk.Label(self, text='')
            # self.label_error.config(font=('Arial', 9))
            # self.canvas1.create_window(400, 310, window=self.label_error)

        else:
            self.address1 = tk.StringVar(self)
            self.address2 = tk.StringVar(self)

            entery = tk.Entry(self, textvariable=self.address1, width=200)
            self.canvas1.create_window(400, 100, window=entery)

            entery = tk.Entry(self, textvariable=self.address2)
            self.canvas1.create_window(400, 150, window=entery, width=200)

            label = tk.Label(self, text='Start Address:')
            label.config(font=('Arial', 9))
            self.canvas1.create_window(290, 100, window=label)

            label = tk.Label(self, text='Goal Address:')
            label.config(font=('Arial', 9))
            self.canvas1.create_window(290, 150, window=label)

    def getStart(self):
        if self.getter != 'points':
            return self.address1.get()
        l1 = self.latlon0.get().split(',')[0]
        l2 = self.latlon0.get().split(',')[1]
        if l1 == '' or l2 == '':
            return None, None
        f1 = float(self.latlon0.get().split(',')[0])
        f2 = float(self.latlon0.get().split(',')[1])
        return f1, f2

    def getEnd(self):
        if self.getter != 'points':
            return self.address2.get()
        l1 = self.latlon1.get().split(',')[0]
        l2 = self.latlon1.get().split(',')[1]
        if l1 == '' or l2 == '':
            return None, None
        f1 = float(self.latlon1.get().split(',')[0])
        f2 = float(self.latlon1.get().split(',')[1])
        return f1, f2

    def applyAlgorithm(self):
        which = self.choiceAlgo.get()
        dist = self.choiceDist.get()

        if which == 'A*':
            which = 0
        else:
            which = 1

        if dist == 'Sphere':
            dist = calcGreatCircleDistanceOnEarth

        elif dist == 'Euclidean':
            dist = calcEuclideanDistanceOnEarth

        elif dist == 'Manhattan':
            dist = calcManhattanDistanceOnEarth

        elif dist == 'Chebyshev':
            dist = calcChebyshevDistanceOnEarth

        elif dist == 'Octile':
            dist = calcOctileDistanceOnEarth

        self.path, self.time, self.steps, total_weight = self.graph.applyAlgorithm(which, dist)

        count = 0
        path_length = 0
        path_time = 0
        path_street_names = []

        d = self.graph.G.edges
        for p in self.path:
            first_node = [p[0], p[1]]
            if first_node in d:
                edge_data = d[(first_node[0], first_node[1], 0)]
                count += 1
                path_length += edge_data['length']
                path_time += edge_data['travel_time']
                if 'name' in edge_data:
                    path_street_names.append(edge_data['name'])
                else:
                    path_street_names.append('no name')

        print('-- Path length = ', path_length, '[meters]')
        print('-- Path travel time = ', (path_time / 60.0), '[minutes]')
        print('-- Path street names = ', path_street_names)
        print('-- Estimated arrival time = ', (total_weight/60.0), '[minutes]', '\n')
        self.show()


app = OurApp(root)
app.mainloop()

# 32.0141
# 34.7736
# 32.0184
# 34.7761
# 32.0184,34.7741 32.0132,34.7793
