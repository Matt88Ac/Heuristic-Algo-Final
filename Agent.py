import tkinter as tk
from tkinter import END
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


class OurApp(tk.Frame):

    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.winfo_toplevel().title("Final Project in Heuristic and Approximation Algorithms")
        self.path = None
        self.button1 = None
        self.button2 = None
        self.button3 = None
        self.button4 = None
        self.button5 = None
        self.button6 = None
        self.plotter = None
        self.canvas1 = tk.Canvas(self, width=800, height=300)
        self.canvas1.pack()

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
        label = tk.Label(self, text='Metric:')
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
        self.graph = RoadMap(start, end)

    def __clearWorld(self):
        self.plotter.get_tk_widget().pack_forget()
        self.path = None
        self.Figure = Figure(dpi=100)
        self.ax = self.Figure.add_subplot(111)

    def show(self):
        if self.plotter:
            self.plotter.get_tk_widget().pack_forget()

        self.Figure = Figure(dpi=100)
        self.ax: plt.Axes = self.Figure.add_subplot(111)

        self.plotter = FigureCanvasTkAgg(self.Figure, self)
        self.plotter.get_tk_widget().pack()

        if self.path is not None:
            self.ax.set_title(f'time of work = {np.round(self.time, 4)}(s), total steps = {self.steps}')

        self.graph.plot(path=self.path, ax=self.ax, show=False)

    def __ApplySettings(self):
        label = tk.Label(self, text='Real World Path Finder')
        label.config(font=('Arial', 20))
        self.canvas1.create_window(400, 20, window=label)

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

    def getStart(self):
        l1 = self.lat0.get()
        l2 = self.lon0.get()
        if l1 == '' or l2 == '':
            return None, None
        return float(self.lat0.get()), float(self.lon0.get())

    def getEnd(self):
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


# canvas1 = tk.Canvas(root, width=800, height=300)
# canvas1.pack()
#
# choiceVar = tk.StringVar()
# choices = ("Sphere", "Euclidean", "Manhattan", "Chebyshev", "Octile")
# choiceVar.set(choices[0])
#
# metrices = [calcGreatCircleDistanceOnEarth, calcEuclideanDistanceOnEarth, calcManhattanDistanceOnEarth,
#            calcChebyshevDistanceOnEarth, calcOctileDistanceOnEarth]
#
# distances = tk.OptionMenu(root, choiceVar, *choices)
# distances.pack()
# choiceVar1 = tk.StringVar()
#
# choices = ('A*', 'Dijkstra')
# choiceVar1.set(choices[0])
# algorithms = tk.OptionMenu(root, choiceVar1, *choices)
# algorithms.pack()
#
# print(choiceVar.get())
#
# canvas1.create_window(50, 220, window=distances)
# canvas1.create_window(140, 220, window=algorithms)
#
# label = tk.Label(root, text='Real World Path Finder')
# label.config(font=('Arial', 20))
# canvas1.create_window(400, 20, window=label)
#
# entry1 = tk.Entry(root)
# entry0 = tk.Entry(root)
#
# label = tk.Label(root, text='Starting Coordinate')
# label.config(font=('Arial', 9))
# canvas1.create_window(400, 80, window=label)
#
# label = tk.Label(root, text='lat:')
# label.config(font=('Arial', 9))
# canvas1.create_window(250, 100, window=label)
# label = tk.Label(root, text='lat:')
# label.config(font=('Arial', 9))
# canvas1.create_window(247, 150, window=label)
#
# label = tk.Label(root, text='lon:')
# label.config(font=('Arial', 9))
# canvas1.create_window(410, 100, window=label)
# label = tk.Label(root, text='lon:')
# label.config(font=('Arial', 9))
# canvas1.create_window(407, 150, window=label)
#
# canvas1.create_window(320, 100, window=entry0)
# canvas1.create_window(480, 100, window=entry1)
#
# entry2 = tk.Entry(root)
# entry3 = tk.Entry(root)
#
# label = tk.Label(root, text='Goal Coordinate')
# label.config(font=('Arial', 9))
# canvas1.create_window(400, 130, window=label)
#
# canvas1.create_window(320, 150, window=entry2)
# canvas1.create_window(480, 150, window=entry3)
#
#
# def create_charts():
#    global x1
#    global x2
#    global x3
#    global bar1
#    global pie2
#    x1 = float(entry1.get())
#    x2 = float(entry2.get())
#    x3 = float(entry3.get())
#
#    figure1 = Figure(dpi=100)
#    subplot1 = figure1.add_subplot(111)
#    xAxis = [float(x1), float(x2), float(x3)]
#    yAxis = [float(x1), float(x2), float(x3)]
#    subplot1.bar(xAxis, yAxis, color='lightsteelblue')
#    bar1 = FigureCanvasTkAgg(figure1, root)
#    bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=0)
#
#    figure2 = Figure(dpi=100)
#    subplot2 = figure2.add_subplot(111)
#    labels2 = 'Label1', 'Label2', 'Label3'
#    pieSizes = [float(x1), float(x2), float(x3)]
#    my_colors2 = ['lightblue', 'lightsteelblue', 'silver']
#    explode2 = (0, 0.1, 0)
#    subplot2.pie(pieSizes, colors=my_colors2, explode=explode2, labels=labels2, autopct='%1.1f%%', shadow=True,
#                 startangle=90)
#    subplot2.axis('equal')
#    pie2 = FigureCanvasTkAgg(figure2, root)
#    pie2.get_tk_widget().pack()
#
#
# def clear_charts():
#    bar1.get_tk_widget().pack_forget()
#    pie2.get_tk_widget().pack_forget()
#
#
# button1 = tk.Button(root, text='  Plot Graph  ', command=create_charts, bg='palegreen2', font=('Arial', 11, 'bold'))
# canvas1.create_window(70, 80, window=button1)
#
# button2 = tk.Button(root, text='  Clear Charts  ', command=clear_charts, bg='lightskyblue2', font=('Arial', 11, 'bold'))
# canvas1.create_window(70, 120, window=button2)
#
# button3 = tk.Button(root, text='Exit Application', command=root.destroy, bg='lightsteelblue2',
#                    font=('Arial', 11, 'bold'))
# canvas1.create_window(70, 160, window=button3)

app = OurApp(root)
app.mainloop()
32.0141
34.7736
32.0184
34.7761
