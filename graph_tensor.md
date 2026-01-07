This is the "Dirty Work" of AI engineering. Most tutorials give you clean CSV files. In reality, OpenStreetMap (OSM) data is messy, huge, and full of strings like "highway": "primary".
To feed this into your GAT-Mamba model, we need to convert the OSM XML map into two specific PyTorch tensors:
 * x (Node Features): A matrix of numbers describing every intersection.
 * edge_index (Connectivity): Who connects to whom.
 * edge_attr (Edge Features): Crucial for traffic. This holds info like "Speed Limit" and "Number of Lanes."
We will use the library osmnx (the gold standard for this).
The Pipeline: From City to Tensors
 * Extract: Download the raw graph for a location (e.g., "Bangalore, India") or load a local .osm file.
 * Simplify: OSM has nodes for every slight curve in a road. We only want Intersections. osmnx does this automatically.
 * Vectorize: Convert text tags ("residential", "one_way") into numbers (One-Hot Encoding).
 * Tensorize: Pack it into a torch_geometric.data.Data object.
The Code Implementation
You will need: pip install osmnx torch torch-geometric
1. The Helper Function (String to Number)
Neural networks can't read "Primary Road." We must convert these to categories.
import torch
import osmnx as ox
import pandas as pd
import numpy as np
from torch_geometric.data import Data

# Define the categories we care about (The "Vocabulary")
ROAD_TYPES = ['residential', 'secondary', 'primary', 'tertiary', 'motorway', 'trunk']

def one_hot_road_type(road_type_string):
    """Converts 'primary' into [0, 0, 1, 0, 0, 0]"""
    # OSM sometimes returns lists for road types, handle that
    if isinstance(road_type_string, list):
        road_type_string = road_type_string[0]
        
    encoding = [0] * len(ROAD_TYPES)
    try:
        idx = ROAD_TYPES.index(road_type_string)
        encoding[idx] = 1
    except ValueError:
        pass # If unknown type, leave as all zeros
    return encoding

2. The Converter (Map to PyTorch)
def osm_to_pytorch(place_name="Koramangala, Bangalore, India"):
    print(f"1. Downloading map for {place_name}...")
    # 'drive' network excludes walking paths
    G = ox.graph_from_place(place_name, network_type='drive')
    
    # ox.graph_to_gdfs returns two Pandas DataFrames: Nodes and Edges
    nodes_df, edges_df = ox.graph_to_gdfs(G)
    
    print(f"   Found {len(nodes_df)} intersections and {len(edges_df)} roads.")

    # --- PART A: NODE FEATURES (x) ---
    # For a traffic map, intersections don't have many intrinsic features 
    # other than position. Let's use Lat/Lon as basic features.
    # In a real app, you might add "Traffic Light: Yes/No" here.
    
    node_features = []
    # We need a mapping from OSM ID (huge number) to Index (0, 1, 2...)
    osm_id_to_index = {osm_id: i for i, osm_id in enumerate(nodes_df.index)}
    
    for osm_id, row in nodes_df.iterrows():
        # Feature: [Latitude, Longitude] (Normalized usually, but raw for now)
        lat = row['y']
        lon = row['x']
        node_features.append([lat, lon])
        
    x = torch.tensor(node_features, dtype=torch.float)

    # --- PART B: EDGE INDICES & ATTRIBUTES ---
    source_nodes = []
    target_nodes = []
    edge_attributes = []

    # Iterate over every road (edge)
    for u, v, key, data in G.edges(keys=True, data=True):
        # 1. Connectivity
        src_idx = osm_id_to_index[u]
        dst_idx = osm_id_to_index[v]
        
        source_nodes.append(src_idx)
        target_nodes.append(dst_idx)
        
        # 2. Edge Features (The Road Info)
        # Extract metadata
        lanes = float(data.get('lanes', 1)) # Default to 1 lane if missing
        speed_limit = data.get('maxspeed', 30) # Default 30 km/h
        
        # Clean speed limit (sometimes it's a list or string like "30 mph")
        if isinstance(speed_limit, list): speed_limit = speed_limit[0]
        try:
            speed_limit = float(str(speed_limit).split()[0])
        except:
            speed_limit = 30.0
            
        # One-Hot Encode the road type
        road_type_vec = one_hot_road_type(data.get('highway', 'residential'))
        
        # Combine: [Lanes, SpeedLimit, ...RoadTypeVector]
        edge_feat = [lanes, speed_limit] + road_type_vec
        edge_attributes.append(edge_feat)

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    edge_attr = torch.tensor(edge_attributes, dtype=torch.float)

    # --- FINAL PYTORCH OBJECT ---
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

# Run it
graph_data = osm_to_pytorch()
print("\n--- Final PyTorch Geometric Object ---")
print(graph_data)
# Expected Output: Data(x=[N, 2], edge_index=[2, E], edge_attr=[E, 8])

Critical Engineering Note for "Project Newton"
You might notice a problem.
 * GAT primarily updates Node features.
 * Traffic actually happens on the Edges (Roads).
If your GNN only looks at Nodes (Intersections), it misses the most important data (Is the road between them jammed?).
Two solutions:
 * The "Line Graph" Conversion: You flip the graph. Roads become Nodes, and Intersections become Edges. This is very popular for traffic.
 * Edge-Aware GAT: You modify the GAT code to include edge_attr in the attention calculation.
   * Standard GAT: Score = Attention(Node A, Node B)
   * Edge GAT: Score = Attention(Node A, Node B, Road Features)
Since you are building a custom engine, Option 2 is better. It keeps the map intuitive (Node=Intersection).
What just happened?
 * We downloaded a real city map.
 * We stripped away the XML junk.
 * We created a Graph Tensor ready for your GAT-Mamba model.
Next Step options:
 * Now that we have the Map (Space), do you want to see how to simulate Traffic Data (Time) on this map so we can actually train the model? (Since we don't have real sensor data yet).
 * Or do you want to see how to Export this entire pipeline (Model + Graph) to C++ for the offline app?
