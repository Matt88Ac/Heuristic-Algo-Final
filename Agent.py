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
        self.button3 = None

        self.canvas1 = tk.Canvas(self, width=800, height=300)
        self.canvas1.pack()
        self.getter = 'points'

        self.button4 = None
        self.button5 = None
        self.button6 = None
        self.plotter = None

        self.defAlgorithms()

        self.__ApplySettings()
        self.create_widgets()
        self.graph = RoadMap

    def create_widgets(self):
        self.button1 = tk.Button(self, text='  Set Road  ', command=self.__setWorld, bg='palegreen2',
                                 font=('Arial', 11, 'bold'))
        self.canvas1.create_window(70, 40, window=self.button1)

        self.button2 = tk.Button(self, text='Apply Algorithm', command=self.applyAlgorithm, bg='gold2',
                                 font=('Arial', 11, 'bold'))
        self.canvas1.create_window(70, 90, window=self.button2)

        self.button3 = tk.Button(self, text='  Plot Graph  ', command=self.show, bg='lightsteelblue2',
                                 font=('Arial', 11, 'bold'))
        self.canvas1.create_window(70, 140, window=self.button3)

        self.button4 = tk.Button(self, text='  Clear Graph  ', command=self.__clearWorld, bg='plum4',
                                 font=('Arial', 11, 'bold'))
        self.canvas1.create_window(70, 190, window=self.button4)

        self.button5 = tk.Button(self, text='Exit Application', command=self.master.destroy, bg='red',
                                 font=('Arial', 11, 'bold'))
        self.canvas1.create_window(70, 240, window=self.button5)

    def defAlgorithms(self):
        self.type = tk.StringVar()
        choices = ('points', 'address')
        self.type.set(self.getter)
        self.typer = tk.OptionMenu(self, self.type, *choices)
        self.typer.pack()
        self.canvas1.create_window(750, 100, window=self.typer)

        label = tk.Label(self, text='Input type:')
        label.config(font=('Arial', 9))
        self.canvas1.create_window(750, 70, window=label)

        label = tk.Label(self, text='Metric Space:')
        label.config(font=('Arial', 9))
        self.canvas1.create_window(650, 190, window=label)

        label = tk.Label(self, text='Algorithm:')
        label.config(font=('Arial', 9))
        self.canvas1.create_window(740, 190, window=label)

        self.choiceDist = tk.StringVar()
        choices = ("Sphere", "Euclidean", "Manhattan", "Chebyshev", "Octile")
        self.choiceDist.set(choices[0])

        self.distances = tk.OptionMenu(self, self.choiceDist, *choices)
        self.distances.pack()
        self.canvas1.create_window(650, 220, window=self.distances)

        self.choiceAlgo = tk.StringVar()
        choices = ('A*', 'Dijkstra')
        self.choiceAlgo.set(choices[0])
        self.algorithms = tk.OptionMenu(self, self.choiceAlgo, *choices)
        self.algorithms.pack()
        self.canvas1.create_window(740, 220, window=self.algorithms)

    def __setWorld(self):
        start = self.getStart()
        end = self.getEnd()
        if start[0] is None or start[1] is None or end[0] is None or end[1] is None:
            raise ValueError
        if self.getter != 'points':
            self.graph = RoadMap(start, end, graph_type='address')
        else:
            self.graph = RoadMap(start, end)

    def __clearWorld(self):
        if self.getter != self.type.get():
            self.getter = self.type.get()
            self.canvas1.delete('all')
            self.defAlgorithms()
            self.create_widgets()
            self.__ApplySettings()
        if self.plotter:
            self.plotter.get_tk_widget().pack_forget()
        self.path = None
        self.Figure = Figure(figsize=(6, 6), dpi=100)
        self.ax = self.Figure.add_subplot(111)

    def show(self):
        if self.plotter:
            self.plotter.get_tk_widget().pack_forget()

        self.Figure = Figure(figsize=(6, 6), dpi=100)
        self.ax: plt.Axes = self.Figure.add_subplot(111)

        self.plotter = FigureCanvasTkAgg(self.Figure, self)
        self.plotter.get_tk_widget().pack()

        if self.path is not None:
            self.ax.set_title(f'time of work = {np.round(self.time, 4)}(s), total steps = {self.steps}')

        try:
            self.graph.plot(path=self.path, ax=self.ax, show=False)

        except TypeError:
            self.ax.set_title('No graph to show.')

    def __ApplySettings(self):
        label = tk.Label(self, text='Real World Path Finder')
        label.config(font=('Arial', 20))
        self.canvas1.create_window(400, 20, window=label)
        if self.type.get() == 'points':
            self.lon1 = tk.StringVar(self)
            self.lat1 = tk.StringVar(self)

            self.lon0 = tk.StringVar(self)
            self.lat0 = tk.StringVar(self)

            entery = tk.Entry(self, textvariable=self.lat0)

            self.canvas1.create_window(320, 100, window=entery)

            entery = tk.Entry(self, textvariable=self.lon0)
            self.canvas1.create_window(480, 100, window=entery)

            entery = tk.Entry(self, textvariable=self.lat1)
            self.canvas1.create_window(320, 150, window=entery)
            entery = tk.Entry(self, textvariable=self.lon1)
            self.canvas1.create_window(480, 150, window=entery)

            label = tk.Label(self, text='Goal Coordinate')
            label.config(font=('Arial', 9))
            self.canvas1.create_window(400, 130, window=label)

            label = tk.Label(self, text='lat:')
            label.config(font=('Arial', 9))
            self.canvas1.create_window(245, 100, window=label)

            label = tk.Label(self, text='Start Coordinate')
            label.config(font=('Arial', 9))
            self.canvas1.create_window(400, 80, window=label)

            label = tk.Label(self, text='lat:')
            label.config(font=('Arial', 9))
            self.canvas1.create_window(245, 150, window=label)

            label = tk.Label(self, text='lon:')

            label.config(font=('Arial', 9))
            self.canvas1.create_window(405, 100, window=label)

            label = tk.Label(self, text='lon:')
            label.config(font=('Arial', 9))
            self.canvas1.create_window(405, 150, window=label)

        else:
            self.address1 = tk.StringVar(self)
            self.address2 = tk.StringVar(self)

            entery = tk.Entry(self, textvariable=self.address1)

            self.canvas1.create_window(400, 100, window=entery)

            entery = tk.Entry(self, textvariable=self.address2)
            self.canvas1.create_window(400, 150, window=entery)

            label = tk.Label(self, text='Start Address:')
            label.config(font=('Arial', 9))
            self.canvas1.create_window(290, 100, window=label)

            label = tk.Label(self, text='Goal Address:')
            label.config(font=('Arial', 9))
            self.canvas1.create_window(290, 150, window=label)

    def getStart(self):
        if self.getter != 'points':
            return self.address1.get()
        l1 = self.lat0.get()
        l2 = self.lon0.get()
        if l1 == '' or l2 == '':
            return None, None
        return float(self.lat0.get()), float(self.lon0.get())

    def getEnd(self):
        if self.getter != 'points':
            return self.address2.get()
        l1 = self.lat1.get()
        l2 = self.lon1.get()
        if l1 == '' or l2 == '':
            return None, None
        return float(self.lat1.get()), float(self.lon1.get())

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

        self.path, self.time, self.steps = self.graph.applyAlgorithm(which, dist)


app = OurApp(root)
app.mainloop()
32.0141
34.7736
32.0184
34.7761
(32.0184, 34.7741), (32.0132, 34.7793)