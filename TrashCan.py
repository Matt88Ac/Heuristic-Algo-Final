import networkx as nx
import matplotlib.pyplot as plt
import osmnx as ox


def print_graph(graph):
    """ prints the graph"""

    # stores the nodes and their name attributes in a dictionary
    nodes_names = nx.get_node_attributes(graph, "name")
    plt.ion()
    pos = nx.spring_layout(graph)

    # draw without labels, cuz it would label them with their adress, since we
    nx.draw(graph, pos, with_labels=False)

    # draw the label with the nodes_names containing the name attribute
    labels = nx.draw_networkx_labels(graph, pos, nodes_names)
    plt.show()


def setup_sending(graph):
    print_graph(graph)

    ###some code doing calculations....

    input('Press enter to continue')


G = ox.graph_from_point((32.0141, 34.7736), dist=200, network_type='walk')
setup_sending(G)
