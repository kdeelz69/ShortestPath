import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import math
import random

# --- RealMap Class ---
class RealMap:
    def __init__(self, G):
        self._graph = G
        self.intersections = {node: (data["x"], data["y"]) for node, data in G.nodes(data=True)}
        self.roads = [list(G.neighbors(node)) for node in G.nodes()]

# --- Load Real Map from GPS Center ---
def load_real_map(center_point=(6.9271, 79.8612), dist=2000):
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

# --- Evaluate Time & Accuracy ---
def evaluate_path_accuracy(map_obj, planner_with_delays, planner_without_delays, traffic_lights):
    def compute_total_time(path, lights):
        total_distance = 0
        total_delay = sum(lights.get(n, 0) for n in path)
        for i in range(len(path) - 1):
            n1, n2 = path[i], path[i + 1]
            x1, y1 = map_obj.intersections[n1]
            x2, y2 = map_obj.intersections[n2]
            total_distance += math.hypot(x2 - x1, y2 - y1) * 111
        speed_kmph = 30
        speed_kmps = speed_kmph / 3600
        time_sec = (total_distance / speed_kmps) + total_delay
        return total_distance, time_sec, total_delay

    dist_delay, time_delay, delay_time = compute_total_time(planner_with_delays.path, traffic_lights)
    dist_base, time_base, _ = compute_total_time(planner_without_delays.path, {n: 0 for n in traffic_lights})

    print("\n📊 Route Comparison Accuracy:")
    print(f"🟥 Delay-Aware Time   : {time_delay/60:.2f} min | Distance: {dist_delay:.2f} km | Delay: {delay_time:.0f} sec")
    print(f"🟦 Distance-Only Time : {time_base/60:.2f} min | Distance: {dist_base:.2f} km")

    diff = time_base - time_delay
    if diff > 0:
        print(f"✅ Delay-aware path is faster by {diff:.2f} seconds.")
    else:
        print(f"⚠️ Distance-only path is faster by {-diff:.2f} seconds.")

# --- Compare Routes on Same Map ---
def show_comparison_map(map_obj, path_delay, path_nodelay, start, goal):
    G = map_obj._graph
    pos = map_obj.intersections

    plt.figure(figsize=(12, 10))
    nx.draw(G, pos, node_size=10, node_color='lightgray', edge_color='whitesmoke', with_labels=False)

    # Start/Goal Markers
    x_start, y_start = pos[start]
    x_goal, y_goal = pos[goal]
    plt.plot(x_start, y_start, 'go', markersize=12, label="Start")
    plt.plot(x_goal, y_goal, 'black', markersize=12, label="Goal")

    # Delay-Aware Route (Red)
    if path_delay:
        delay_edges = list(zip(path_delay[:-1], path_delay[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=delay_edges, edge_color='red', width=3, label="Delay-Aware Path")

    # Distance-Only Route (Blue, Dashed)
    if path_nodelay:
        nodelay_edges = list(zip(path_nodelay[:-1], path_nodelay[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=nodelay_edges, edge_color='blue', style='dashed', width=2.5, label="Distance-Only Path")

    plt.title("🗺️ Route Comparison: Delay-Aware vs Distance-Only")
    plt.legend()
    plt.axis("off")
    plt.show()

# --- Utility to Get Nearest Node ---
def get_nearest_node(G, lat, lon):
    return ox.distance.nearest_nodes(G, X=lon, Y=lat)

# --- Main Runner ---
if __name__ == "__main__":
    center = (6.9271, 79.8612)  # Colombo City
    map_obj, raw_graph = load_real_map(center_point=center, dist=2000)

    start_coords = (6.9275, 79.8600)
    goal_coords = (6.9305, 79.8720)

    start_node = get_nearest_node(raw_graph, *start_coords)
    goal_node = get_nearest_node(raw_graph, *goal_coords)

    # Simulate traffic light delays
    traffic_lights = {node: 0 for node in raw_graph.nodes()}
    for node in list(raw_graph.nodes())[::15]:
        traffic_lights[node] = random.randint(5, 25)

    # Run both planners
    planner_with = PathPlanner(map_obj, start=start_node, goal=goal_node, traffic_lights=traffic_lights)
    planner_without = PathPlanner(map_obj, start=start_node, goal=goal_node, traffic_lights={n: 0 for n in traffic_lights})

    if planner_with.path and planner_without.path:
        print("✅ Both routes computed successfully.")
        show_comparison_map(map_obj, planner_with.path, planner_without.path, start_node, goal_node)
        evaluate_path_accuracy(map_obj, planner_with, planner_without, traffic_lights)
    else:
        print("❌ One or both routes could not be found.")
