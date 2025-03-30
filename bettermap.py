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

def calculate_total_distance_and_time(map_obj, path, traffic_lights, speed_kmph=30):
    total_distance = 0
    total_delay = 0

    for i in range(len(path) - 1):
        node1 = path[i]
        node2 = path[i + 1]
        lat1, lon1 = map_obj.intersections[node1]
        lat2, lon2 = map_obj.intersections[node2]
        dist = math.hypot(lon2 - lon1, lat2 - lat1) * 111  # Approx degrees to km
        total_distance += dist

    # Total delay in seconds
    total_delay = sum(traffic_lights.get(node, 0) for node in path)

    # Convert speed to km per second
    speed_kmps = speed_kmph / 3600.0

    # Time = distance / speed
    time_without_delay = total_distance / speed_kmps  # in seconds
    total_time_sec = time_without_delay + total_delay

    return total_distance, total_time_sec, total_delay


# --- Show Route on Interactive Map ---
from shapely.geometry import LineString
import geopandas as gpd

def show_interactive_map(raw_graph, path, start_coords, goal_coords, traffic_lights, save_as="shortest_route_map.html"):
    # Convert graph to GeoDataFrames in lat/lon
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(raw_graph, nodes=True, edges=True)

    # Extract path edges
    path_edges = list(zip(path[:-1], path[1:]))

    # Build route geometry from edge geometries
    route_lines = []
    for u, v in path_edges:
        try:
            edge_data = raw_graph.get_edge_data(u, v)
            if edge_data:
                geom = edge_data[0].get("geometry", None)
                if geom:
                    route_lines.append(geom)
                else:
                    # Use straight line if no geometry
                    point_u = (raw_graph.nodes[u]['x'], raw_graph.nodes[u]['y'])
                    point_v = (raw_graph.nodes[v]['x'], raw_graph.nodes[v]['y'])
                    route_lines.append(LineString([point_u, point_v]))
        except:
            continue

    # Merge all segments into one LineString
    full_route = gpd.GeoSeries(route_lines).unary_union

    # Center folium map on start location
    m = folium.Map(location=start_coords, zoom_start=15)

    # Plot route
    if full_route:
        if full_route.geom_type == "MultiLineString":
            for line in full_route.geoms:
                folium.PolyLine([(lat, lon) for lon, lat in line.coords], color="red", weight=6).add_to(m)
        else:
            folium.PolyLine([(lat, lon) for lon, lat in full_route.coords], color="red", weight=6).add_to(m)

    # Add traffic light markers
    for node in path:
        lat, lon = raw_graph.nodes[node]['y'], raw_graph.nodes[node]['x']
        delay = traffic_lights.get(node, 0)
        folium.CircleMarker(
            location=(lat, lon),
            radius=4,
            color="orange" if delay > 0 else "green",
            fill=True,
            popup=f"Node: {node}<br>Delay: {delay}s"
        ).add_to(m)

    # Add start & goal
    folium.Marker(start_coords, popup="Start", icon=folium.Icon(color="blue")).add_to(m)
    folium.Marker(goal_coords, popup="Goal", icon=folium.Icon(color="red")).add_to(m)

    # Show route summary
    total_km, total_sec, delay_sec = calculate_total_distance_and_time(map_obj, path, traffic_lights)
    summary_html = f"""
    <div style="font-size: 14px; background: white; padding: 4px;">
        <b>Route Summary</b><br>
        Distance: {total_km:.2f} km<br>
        Est. Time: {total_sec/60:.2f} min<br>
        Delay: {delay_sec:.0f} sec
    </div>
    """
    mid_index = len(path) // 2
    mid_lat = raw_graph.nodes[path[mid_index]]['y']
    mid_lon = raw_graph.nodes[path[mid_index]]['x']
    folium.Marker(
        location=(mid_lat, mid_lon),
        icon=folium.DivIcon(html=summary_html)
    ).add_to(m)

    # Save and notify
    m.save(save_as)
    print(f"✅ Map saved: {save_as}")



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
        show_interactive_map(raw_graph, planner.path, start_coords, goal_coords, traffic_lights)

    else:
        print("❌ No path could be found.")
