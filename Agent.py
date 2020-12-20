import osmnx as ox
import networkx as nx

start = (32.0141, 34.7745)
end = (32.0148, 34.7736)


G = ox.graph_from_point((32.0141, 34.7736), dist=dist, network_type='walk')

start_node = ox.get_nearest_node(G, start)
end_node = ox.get_nearest_node(G, end)
G.add_node(start_node)
G.add_node(end_node)

#ox.plot_graph(G, node_color='blue', bgcolor='white', edge_color='k')


route2 = nx.shortest_path(G, start_node, end_node, weight='travel_time')
print(route2)
#route3 = nx.shortest_path(G, start_node[0], end_node, weight='travel_time')

#ox.plot_graph_route(G, route1, route_linewidth=6, node_size=0, bgcolor='k')
##ox.plot_graph_route(G, route2, route_linewidth=6, node_size=0, bgcolor='k')
#ox.plot_graph_route(G, route3, route_linewidth=6, node_size=0, bgcolor='k')
