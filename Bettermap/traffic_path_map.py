import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import math

# --- RealMap Class ---
class RealMap:
    def __init__(self, G):
        self._graph = G
        self.intersections = {node: (data["x"], data["y"]) for node, data in G.nodes(data=True)}
        self.roads = [list(G.neighbors(node)) for node in G.nodes()]

# --- Load Real Map from GPS Center ---
def load_real_map(center_point=(9.6615, 80.0255), dist=2000):
    G = ox.graph_from_point(center_point, dist=dist, network_type="drive")
    return RealMap(G), G

# --- PathPlanner with Traffic Light Delays ---
class PathPlanner():
    def __init__(self, M, start=None, goal=None, heuristic_weight=1.0, traffic_lights=None):
        self.map = M
        self.start = start
        self.goal = goal
        self.heuristic_weight = heuristic_weight
        self.traffic_lights = traffic_lights if traffic_lights else {node: 0 for node in M.intersections}
        if self.start is not None and self.goal is not None:
            self._reset()

    def _reset(self):
        self.closedSet = set()
        self.openSet = {self.start}
        self.cameFrom = {}
        self.gScore = {node: float('inf') for node in self.map.intersections}
        self.gScore[self.start] = 0
        self.fScore = {node: float('inf') for node in self.map.intersections}
        self.fScore[self.start] = self.heuristic_cost_estimate(self.start)
        self.path = self.run_search()

    def run_search(self):
        while self.openSet:
            current = min(self.openSet, key=lambda node: self.fScore[node])
            if current == self.goal:
                return list(reversed(self.reconstruct_path(current)))
            self.openSet.remove(current)
            self.closedSet.add(current)
            for neighbor in self.map._graph.neighbors(current):
                if neighbor in self.closedSet:
                    continue
                tentative_g = self.gScore[current] + self.distance(current, neighbor) + self.traffic_lights.get(neighbor, 0)
                if tentative_g < self.gScore[neighbor]:
                    self.cameFrom[neighbor] = current
                    self.gScore[neighbor] = tentative_g
                    self.fScore[neighbor] = tentative_g + self.heuristic_cost_estimate(neighbor)
                    self.openSet.add(neighbor)
        print("No Path Found")
        return None

    def reconstruct_path(self, current):
        path = [current]
        while current in self.cameFrom:
            current = self.cameFrom[current]
            path.append(current)
        return path

    def distance(self, a, b):
        x1, y1 = self.map.intersections[a]
        x2, y2 = self.map.intersections[b]
        return math.hypot(x2 - x1, y2 - y1)

    def heuristic_cost_estimate(self, node):
        x1, y1 = self.map.intersections[node]
        x2, y2 = self.map.intersections[self.goal]
        return self.heuristic_weight * math.hypot(x2 - x1, y2 - y1)

# --- Visualization with Delays ---
def show_map(map_obj, path=None, start=None, goal=None, traffic_lights=None):
    G = map_obj._graph
    pos = map_obj.intersections

    # Define node colors
    node_colors = []
    for node in G.nodes():
        if node == start:
            node_colors.append("blue")  # Start
        elif node == goal:
            node_colors.append("red")   # Goal
        elif traffic_lights and traffic_lights.get(node, 0) > 0:
            node_colors.append("orange")  # Delay nodes
        elif path and node in path:
            node_colors.append("green")   # Path
        else:
            node_colors.append("gray")    # Other

    plt.figure(figsize=(12, 10))
    nx.draw(G, pos, node_size=30, node_color=node_colors, edge_color='lightgray', with_labels=False)

    # Highlight route edges
    if path:
        route_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=route_edges, edge_color='red', width=2)

    plt.title("A* Route with Traffic Light Delays")
    plt.axis("off")
    plt.show()

# --- Utility to Get Nearest Node from GPS ---
def get_nearest_node(G, lat, lon):
    return ox.distance.nearest_nodes(G, X=lon, Y=lat)

# --- Main Runner ---
if __name__ == "__main__":
    center = (9.6615, 80.0255)  # Jaffna center
    map_obj, raw_graph = load_real_map(center_point=center, dist=2000)

    # Define GPS Start and Goal
    start_coords = (9.6680, 80.0115)  # Jaffna Teaching Hospital
    goal_coords = (9.6639, 80.0256)   # Nallur Kandaswamy Temple

    # Get nearest nodes
    start_node = get_nearest_node(raw_graph, *start_coords)
    goal_node = get_nearest_node(raw_graph, *goal_coords)

    # Simulate delays
    traffic_lights = {node: 0 for node in raw_graph.nodes()}
    traffic_lights[start_node] = 0
    traffic_lights[goal_node] = 0
    # Add some sample delays
    import random
    for node in list(raw_graph.nodes())[::10]:  # Random every 10th node
        traffic_lights[node] = random.randint(5, 20)

    # Run A* path planner
    planner = PathPlanner(map_obj, start=start_node, goal=goal_node, heuristic_weight=1.0, traffic_lights=traffic_lights)

    if planner.path:
        print("✅ Path found!")
        show_map(map_obj, path=planner.path, start=start_node, goal=goal_node, traffic_lights=traffic_lights)
    else:
        print("❌ No path could be found.")
