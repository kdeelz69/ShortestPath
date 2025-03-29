import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import math

# --- RealMap Class Compatible with PathPlanner ---
class RealMap:
    def __init__(self, G):
        self._graph = G
        self.intersections = {node: (data["x"], data["y"]) for node, data in G.nodes(data=True)}
        self.roads = [list(G.neighbors(node)) for node in G.nodes()]

# --- Load Real Map from GPS Point (not place name) ---
def load_real_map(center_point=(9.6615, 80.0255), dist=2000):
    G = ox.graph_from_point(center_point, dist=dist, network_type="drive")
   
    return RealMap(G), G

# --- PathPlanner Class (A* with traffic light delays) ---
class PathPlanner():
    def __init__(self, M, start=None, goal=None, heuristic_weight=1.0, traffic_lights=None):
        self.map = M
        self.start = start
        self.goal = goal
        self.heuristic_weight = heuristic_weight
        self.traffic_lights = traffic_lights if traffic_lights is not None else {node: 0 for node in M.intersections}
        if self.start is not None and self.goal is not None:
            self._reset()

    def reconstruct_path(self, current):
        total_path = [current]
        while current in self.cameFrom:
            current = self.cameFrom[current]
            total_path.append(current)
        return total_path

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
                tentative_gScore = self.gScore[current] + self.distance(current, neighbor) + self.traffic_lights.get(neighbor, 0)
                if tentative_gScore < self.gScore[neighbor]:
                    self.cameFrom[neighbor] = current
                    self.gScore[neighbor] = tentative_gScore
                    self.fScore[neighbor] = tentative_gScore + self.heuristic_cost_estimate(neighbor)
                    self.openSet.add(neighbor)
        print("No Path Found")
        return None

    def distance(self, node1, node2):
        x1, y1 = self.map.intersections[node1]
        x2, y2 = self.map.intersections[node2]
        return math.hypot(x2 - x1, y2 - y1)

    def heuristic_cost_estimate(self, node):
        if self.goal is None:
            return 0
        x1, y1 = self.map.intersections[node]
        x2, y2 = self.map.intersections[self.goal]
        return self.heuristic_weight * math.hypot(x2 - x1, y2 - y1)

# --- Show Map ---
def show_map(map_obj, path=None, start=None, goal=None):
    G = map_obj._graph
    fig, ax = ox.plot_graph(G, node_size=10, show=False, close=False)

    if path:
        ox.plot_graph_route(G, path, route_color='green', route_linewidth=4, node_size=0, ax=ax, show=False, close=False)

    if start:
        x, y = map_obj.intersections[start]
        ax.plot(x, y, 'bo', markersize=30, label='Start')
    if goal:
        x, y = map_obj.intersections[goal]
        ax.plot(x, y, 'ro', markersize=30, label='Goal')

    plt.legend()
    plt.title("Route on Real Map")
    plt.show()

# --- Utility to Get Nearest Node from GPS ---
def get_nearest_node(G, lat, lon):
    return ox.distance.nearest_nodes(G, X=lon, Y=lat)

# --- Main Runner ---
if __name__ == "__main__":
    # Load real map around this location in Jaffna
    center = (9.6615, 80.0255)
    map_obj, raw_graph = load_real_map(center_point=center, dist=2000)

    # Define start and goal GPS coordinates
    start_coords = (9.6615, 80.0255)
    goal_coords = (9.6695, 80.0180)

    # Get nearest nodes on the graph
    start_node = get_nearest_node(raw_graph, *start_coords)
    goal_node = get_nearest_node(raw_graph, *goal_coords)

    # Simulate traffic lights (optional)
    traffic_lights = {
        start_node: 0,
        goal_node: 0,
        # Add other node delays as needed, e.g.:
        # some_other_node_id: 15,
    }

    # Run the A* path planner
    planner = PathPlanner(map_obj, start=start_node, goal=goal_node, heuristic_weight=1.0, traffic_lights=traffic_lights)

    # Show the result
    if planner.path:
        print("Path found!")
        show_map(map_obj, path=planner.path, start=start_node, goal=goal_node)
    else:
        print("No path could be found.")
