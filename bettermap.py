import osmnx as ox
import networkx as nx
import folium
import math

# --- RealMap Class Compatible with PathPlanner ---
class RealMap:
    def __init__(self, G):
        self._graph = G
        self.intersections = {node: (data["y"], data["x"]) for node, data in G.nodes(data=True)}  # (lat, lon)
        self.roads = [list(G.neighbors(node)) for node in G.nodes()]

# --- Load Real Map from GPS Point ---
def load_real_map(center_point=(9.6615, 80.0255), dist=2000):
    G = ox.graph_from_point(center_point, dist=dist, network_type="drive")
    G = ox.project_graph(G)
    return RealMap(G), G

# --- PathPlanner Class (A* with traffic light delays) ---
class PathPlanner:
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
        x1, y1 = self.map.intersections[node1][1], self.map.intersections[node1][0]
        x2, y2 = self.map.intersections[node2][1], self.map.intersections[node2][0]
        return math.hypot(x2 - x1, y2 - y1)

    def heuristic_cost_estimate(self, node):
        if self.goal is None:
            return 0
        x1, y1 = self.map.intersections[node][1], self.map.intersections[node][0]
        x2, y2 = self.map.intersections[self.goal][1], self.map.intersections[self.goal][0]
        return self.heuristic_weight * math.hypot(x2 - x1, y2 - y1)

# --- Utility to Get Nearest Node from GPS ---
def get_nearest_node(G, lat, lon):
    return ox.distance.nearest_nodes(G, X=lon, Y=lat)

# --- Show Route on Interactive Map ---
def show_interactive_map(raw_graph, path, start_coords, goal_coords, save_as="shortest_route_map.html"):
    # Convert graph to geographic coordinates
    G_proj = raw_graph
    G_geo = ox.project_graph(G_proj, to_crs='epsg:4326')  # Ensure graph is in lat/lon

    # Get coordinates of each node in path
    route_nodes = path
    route_coords = [(G_geo.nodes[n]['y'], G_geo.nodes[n]['x']) for n in route_nodes]

    # Create interactive map
    m = folium.Map(location=start_coords, zoom_start=15)

    # Draw route
    folium.PolyLine(route_coords, color="green", weight=5, opacity=0.8).add_to(m)

    # Start/goal markers
    folium.Marker(start_coords, popup="Start", icon=folium.Icon(color="blue")).add_to(m)
    folium.Marker(goal_coords, popup="Goal", icon=folium.Icon(color="red")).add_to(m)

    # Save and show map
    m.save(save_as)
    print(f"✅ Map saved as {save_as}")


# --- Main Runner ---
if __name__ == "__main__":
    # Define map center
    center = (9.6615, 80.0255)
    map_obj, raw_graph = load_real_map(center_point=center, dist=2000)

    # Change these coordinates for different points
    start_coords = (9.6680, 80.0115)  # Example: Jaffna Teaching Hospital
    goal_coords = (9.6639, 80.0256)   # Example: Nallur Kandaswamy Kovil

    # Get nearest nodes from GPS
    start_node = get_nearest_node(raw_graph, *start_coords)
    goal_node = get_nearest_node(raw_graph, *goal_coords)

    # Optional: define traffic light delays
    traffic_lights = {
        start_node: 0,
        goal_node: 0,
        # You can add delays to other nodes if you like
    }

    # Find the shortest path
    planner = PathPlanner(map_obj, start=start_node, goal=goal_node, heuristic_weight=1.0, traffic_lights=traffic_lights)

    if planner.path:
        print("🚗 Path found! Rendering map...")
        show_interactive_map(raw_graph, planner.path, start_coords, goal_coords)

    else:
        print("❌ No path could be found.")
