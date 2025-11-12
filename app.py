import streamlit as st
import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
import collections
import csv
import io
from PIL import Image
import tempfile
import os
import math
import networkx as nx
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Image Graph Generation App",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Generate Graph Data from Image")
st.markdown("Upload an image to perform skeletonization and graph construction, generating CSV data.")

# Sidebar parameter settings
st.sidebar.header("⚙️ Settings")

# Distance scale settings
st.sidebar.subheader("📏 Distance Scale Settings")
enable_distance_scale = st.sidebar.checkbox("Enable real distance calculation", value=False)

if enable_distance_scale:
    st.sidebar.markdown("**Image Range (Latitude/Longitude)**")
    
    col_lat1, col_lat2 = st.sidebar.columns(2)
    with col_lat1:
        north_latitude = st.number_input("North Latitude", value=35.1, format="%.6f", step=0.000001)
    with col_lat2:
        south_latitude = st.number_input("South Latitude", value=35.0, format="%.6f", step=0.000001)
    
    col_lon1, col_lon2 = st.sidebar.columns(2)
    with col_lon1:
        west_longitude = st.number_input("West Longitude", value=135.0, format="%.6f", step=0.000001)
    with col_lon2:
        east_longitude = st.number_input("East Longitude", value=135.1, format="%.6f", step=0.000001)
    
    # Image size (pixels)
    st.sidebar.markdown("**Image Size (Pixels)**")
    col_size1, col_size2 = st.sidebar.columns(2)
    with col_size1:
        image_width_px = st.number_input("Width", value=480, min_value=1)
    with col_size2:
        image_height_px = st.number_input("Height", value=360, min_value=1)

# Image processing settings
st.sidebar.subheader("Image Processing")
resize_enabled = st.sidebar.checkbox("Resize image to 480x360", value=True)

# Graph construction settings
st.sidebar.subheader("Graph Construction")
curvature_threshold = st.sidebar.slider("Curvature split threshold", 1.0, 30.0, 10.0, 0.5)
max_jump_distance = st.sidebar.slider("Max jump distance", 1, 5, 2)
min_intersection_transitions = st.sidebar.slider("Intersection detection threshold", 2, 5, 3)
min_node_area = st.sidebar.slider("Minimum node area", 1, 10, 1)

# Debug mode
st.sidebar.subheader("🐛 Debug")
debug_mode = st.sidebar.checkbox("Enable debug mode", value=True)

# Network integration settings
st.sidebar.subheader("🔗 Network Integration")
enable_integration = st.sidebar.checkbox("Integrate isolated networks", value=True)
if enable_integration:
    integration_threshold = st.sidebar.slider("Integration distance threshold (pixels)", 5, 50, 30, 5)

# File upload
uploaded_file = st.file_uploader("Upload image file", type=['png', 'jpg', 'jpeg'])


# --- Function definitions ---

def calculate_distance_scale(north_lat, south_lat, west_lon, east_lon, width_px, height_px):
    """
    Calculate distance scale from image latitude/longitude range
    
    Parameters:
    - north_lat, south_lat: North and south latitude boundaries
    - west_lon, east_lon: West and east longitude boundaries
    - width_px, height_px: Image width and height (pixels)
    
    Returns:
    - meters_per_pixel_x: Meters per pixel horizontally
    - meters_per_pixel_y: Meters per pixel vertically
    - meters_per_pixel_avg: Average meters per pixel
    """
    # Earth radius (meters)
    EARTH_RADIUS = 6371000
    
    # Calculate center latitude
    center_lat = (north_lat + south_lat) / 2
    center_lat_rad = math.radians(center_lat)
    
    # Longitude difference (east-west distance)
    lon_diff = abs(east_lon - west_lon)
    lon_diff_rad = math.radians(lon_diff)
    distance_x_meters = EARTH_RADIUS * lon_diff_rad * math.cos(center_lat_rad)
    meters_per_pixel_x = distance_x_meters / width_px
    
    # Latitude difference (north-south distance)
    lat_diff = abs(north_lat - south_lat)
    lat_diff_rad = math.radians(lat_diff)
    distance_y_meters = EARTH_RADIUS * lat_diff_rad
    meters_per_pixel_y = distance_y_meters / height_px
    
    # Average value (for diagonal distance calculation)
    meters_per_pixel_avg = (meters_per_pixel_x + meters_per_pixel_y) / 2
    
    return meters_per_pixel_x, meters_per_pixel_y, meters_per_pixel_avg


def resize_image(img, target_width=480, target_height=360):
    """Resize image"""
    original_height, original_width = img.shape[:2]
    resized_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized_img, original_height, original_width


def refine_skeleton_branches(skeleton):
    """Refine skeleton branches"""
    H, W = skeleton.shape
    refined = skeleton.copy()
    
    neighbors_8 = [(-1, -1), (-1, 0), (-1, 1), (0, 1), 
                   (1, 1), (1, 0), (1, -1), (0, -1)]
    
    endpoints = []
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if skeleton[y, x] == 1:
                neighbors = [skeleton[y + dy, x + dx] for dy, dx in neighbors_8]
                neighbor_count = sum(neighbors)
                
                if neighbor_count == 1:
                    endpoints.append((y, x))
    
    min_branch_length = 5
    for start_y, start_x in endpoints:
        branch_length = 0
        y, x = start_y, start_x
        branch_pixels = [(y, x)]
        
        while branch_length < min_branch_length:
            neighbors = []
            for dy, dx in neighbors_8:
                ny, nx = y + dy, x + dx
                if (0 <= ny < H and 0 <= nx < W and 
                    skeleton[ny, nx] == 1 and (ny, nx) not in branch_pixels):
                    neighbors.append((ny, nx))
            
            if len(neighbors) == 0:
                break
            elif len(neighbors) == 1:
                y, x = neighbors[0]
                branch_pixels.append((y, x))
                branch_length += 1
            else:
                break
        
        if branch_length < min_branch_length and len(neighbors) == 0:
            for py, px in branch_pixels:
                refined[py, px] = 0
    
    return refined


def high_quality_skeletonization(img):
    """High-quality skeletonization"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Use adaptive thresholding to extract lines even with non-uniform background
    binary = cv2.adaptiveThreshold(
        denoised, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        blockSize=11, 
        C=2
    )
    
    kernel_small = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
    
    kernel_dilate = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(cleaned, kernel_dilate, iterations=1)
    
    binary_bool = (dilated > 128).astype(bool)
    # Skeletonization
    skeleton_bool = skeletonize(binary_bool)
    skeleton = skeleton_bool.astype(np.uint8)
    
    # Remove components that are too small
    labeled_skeleton = label(skeleton, connectivity=2)
    regions = regionprops(labeled_skeleton)
    
    min_component_size = 5
    filtered_skeleton = np.zeros_like(skeleton)
    for region in regions:
        if region.area >= min_component_size:
            for coord in region.coords:
                filtered_skeleton[coord[0], coord[1]] = 1
    
    # Remove short branches
    filtered_skeleton = refine_skeleton_branches(filtered_skeleton)
    
    processed_img = (filtered_skeleton * 255).astype(np.uint8)
    
    return filtered_skeleton, processed_img


def detect_and_build_graph(binary_img, curvature_threshold, max_jump, min_transitions, min_area, debug=False):
    """Graph detection and construction (with corner detection and debug info)"""
    H, W = binary_img.shape
    
    feature_map = np.zeros_like(binary_img)
    neighbors_coord = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    feature_pixels = {}
    
    # Debug counters
    debug_info = {
        'total_skeleton_pixels': int(binary_img.sum()),
        'intersections_found': 0,
        'corners_found': 0,
        'endpoints_found': 0,
        'feature_pixels_before_clustering': 0,
        'regions_found': 0,
        'regions_kept': 0
    }
    
    # Detect feature points (intersections, corners, and endpoints)
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if binary_img[y, x] == 1:
                neighbors = [(binary_img[y + dy, x + dx]) for dy, dx in neighbors_coord]
                transitions = sum(neighbors[i] == 0 and neighbors[(i + 1) % 8] == 1 for i in range(8))
                
                is_feature = False
                node_type = -1
                
                if transitions >= min_transitions:
                    is_feature = True
                    node_type = 0  # Intersection
                    debug_info['intersections_found'] += 1
                elif transitions == 2:
                    is_feature = True
                    node_type = 1  # Corner
                    debug_info['corners_found'] += 1
                elif transitions == 1:
                    is_feature = True
                    node_type = 2  # Endpoint
                    debug_info['endpoints_found'] += 1
                
                if is_feature:
                    feature_map[y, x] = 1
                    feature_pixels[(y, x)] = node_type
    
    debug_info['feature_pixels_before_clustering'] = len(feature_pixels)
    
    if feature_map.sum() == 0:
        if debug:
            st.warning(f"⚠️ No feature points detected! Debug info: {debug_info}")
        return None, None, None, debug_info
    
    # Label and cluster existing feature pixels
    labeled_img = label(feature_map, connectivity=2)
    regions = regionprops(labeled_img)
    debug_info['regions_found'] = len(regions)
    
    nodes = {}
    coord_to_node_id = np.full((H, W), -1, dtype=int)
    node_id_counter = 1
    
    for region in regions:
        if region.area < min_area:
            continue
        
        debug_info['regions_kept'] += 1
        node_id = node_id_counter
        center_y, center_x = region.centroid
        
        cluster_types = [feature_pixels[(py, px)] for py, px in region.coords if (py, px) in feature_pixels]
        if cluster_types:
            most_common_type = collections.Counter(cluster_types).most_common(1)[0][0]
        else:
            continue
        
        # Map actual node region only
        for y, x in region.coords:
            coord_to_node_id[y, x] = node_id
        
        # Create node data
        nodes[node_id] = {
            'pos': (int(center_x), int(center_y)), 
            'type': most_common_type, 
            'adj': [], 
            'coords': list(region.coords)
        }
        
        node_id_counter += 1
    
    if len(nodes) == 0:
        if debug:
            st.warning(f"⚠️ No nodes created after clustering! Debug info: {debug_info}")
        return None, None, None, debug_info
    
    marked_img = cv2.cvtColor(binary_img * 255, cv2.COLOR_GRAY2BGR)
    edges = set()
    edge_visited_map = np.full((H, W), -1, dtype=int)
    edge_id_counter = 0
    
    # Enumerate edge search starting points
    start_pixels = []
    for node_id, node_data in nodes.items():
        for start_y, start_x in node_data['coords']: 
            for dy, dx in neighbors_coord:
                neighbor_y, neighbor_x = start_y + dy, start_x + dx
                
                if (0 <= neighbor_y < H and 0 <= neighbor_x < W and 
                    binary_img[neighbor_y, neighbor_x] == 1):
                    
                    neighbor_node_id = coord_to_node_id[neighbor_y, neighbor_x]
                    
                    # If adjacent pixel belongs to another node, connect directly
                    if neighbor_node_id != -1 and neighbor_node_id != node_id:
                        n1, n2 = min(node_id, neighbor_node_id), max(node_id, neighbor_node_id)
                        edge_key = (n1, n2)
                        
                        if edge_key not in edges:
                            edges.add(edge_key)
                            
                            # Edge length is 1 (direct contact)
                            if neighbor_node_id not in [adj[0] for adj in nodes[node_id]['adj']]:
                                nodes[node_id]['adj'].append((neighbor_node_id, 1))
                            if node_id not in [adj[0] for adj in nodes[neighbor_node_id]['adj']]:
                                nodes[neighbor_node_id]['adj'].append((node_id, 1))
                    
                    # If adjacent pixel doesn't belong to any node, it's an edge starting point candidate
                    elif neighbor_node_id == -1:
                        start_pixels.append((node_id, start_y, start_x, neighbor_y, neighbor_x))
    
    processed_starts = set()
    
    # Edge search
    for node_id, start_y, start_x, initial_y, initial_x in start_pixels:
        start_key = (node_id, initial_y, initial_x)
        if start_key in processed_starts:
            continue
        if edge_visited_map[initial_y, initial_x] != -1:
            continue
        
        path = []
        temp_path_visited = set()
        y, x = initial_y, initial_x
        prev_dy, prev_dx = initial_y - start_y, initial_x - start_x
        current_curvature = 0.0
        current_start_node_id = node_id
        
        while True:
            # Check if current position belongs to another node
            end_node_id_check = coord_to_node_id[y, x]
            is_end_node = (end_node_id_check != -1 and end_node_id_check != current_start_node_id)
            is_split_point = (current_curvature >= curvature_threshold) and (end_node_id_check == -1)
            
            if is_end_node or is_split_point:
                target_node_id = -1
                if is_end_node:
                    target_node_id = end_node_id_check
                elif is_split_point:
                    target_node_id = node_id_counter
                    nodes[target_node_id] = {
                        'pos': (x, y), 
                        'type': 3, 
                        'adj': [], 
                        'coords': [(y, x)]
                    }
                    coord_to_node_id[y, x] = target_node_id
                    node_id_counter += 1
                
                n1, n2 = min(current_start_node_id, target_node_id), max(current_start_node_id, target_node_id)
                edge_key = (n1, n2)
                
                if current_start_node_id == node_id or edge_key not in edges:
                    edges.add(edge_key)
                    length = len(path)
                    
                    existing_adj = [adj[0] for adj in nodes[current_start_node_id]['adj']]
                    if target_node_id not in existing_adj:
                        nodes[current_start_node_id]['adj'].append((target_node_id, length))

                    existing_adj = [adj[0] for adj in nodes[target_node_id]['adj']]
                    if current_start_node_id not in existing_adj:
                        nodes[target_node_id]['adj'].append((current_start_node_id, length))
                    
                    edge_id_counter += 1
                    for py, px in path:
                        marked_img[py, px] = (0, 255, 0)
                        edge_visited_map[py, px] = edge_id_counter
                
                if is_end_node:
                    break
                elif is_split_point:
                    current_start_node_id = target_node_id
                    current_curvature = 0.0
                    path = []
            
            if edge_visited_map[y, x] != -1:
                break
            
            path.append((y, x))
            temp_path_visited.add((y, x))
            
            best_pixel = None
            best_vector = (0, 0)
            best_score = -2
            
            for dy_search in range(-max_jump, max_jump + 1):
                for dx_search in range(-max_jump, max_jump + 1):
                    if dy_search == 0 and dx_search == 0:
                        continue
                    next_y, next_x = y + dy_search, x + dx_search
                    
                    if not (0 <= next_y < H and 0 <= next_x < W):
                        continue
                    
                    if (next_y, next_x) in temp_path_visited or edge_visited_map[next_y, next_x] != -1:
                         continue
                    
                    if binary_img[next_y, next_x] == 1:
                        # If reached a node, select it preferentially
                        if coord_to_node_id[next_y, next_x] != -1 and coord_to_node_id[next_y, next_x] != current_start_node_id:
                            best_pixel = (next_y, next_x)
                            best_vector = (dy_search, dx_search)
                            best_score = 10
                            break
                            
                        current_vector = (dy_search, dx_search)
                        is_adjacent = max(abs(dy_search), abs(dx_search)) == 1
                        is_jump = max(abs(dy_search), abs(dx_search)) == 2
                        
                        if is_adjacent:
                            score = prev_dy * dy_search + prev_dx * dx_search
                            if score > best_score:
                                best_score = score
                                best_pixel = (next_y, next_x)
                                best_vector = current_vector
                        
                        elif is_jump:
                            mid_y1, mid_x1 = y + dy_search//2, x + dx_search//2
                            if binary_img[mid_y1, mid_x1] == 0:
                                score = prev_dy * dy_search + prev_dx * dx_search - 3
                                if score > best_score:
                                    best_score = score
                                    best_pixel = (next_y, next_x)
                                    best_vector = current_vector
                
                if best_score == 10:
                    break
            
            if best_pixel:
                new_dy, new_dx = best_vector
                
                if coord_to_node_id[best_pixel[0], best_pixel[1]] != -1 and coord_to_node_id[best_pixel[0], best_pixel[1]] != current_start_node_id:
                    y, x = best_pixel
                    prev_dy, prev_dx = new_dy, new_dx
                    continue
                
                curvature_change = 2 - (prev_dy * new_dy + prev_dx * new_dx)
                current_curvature += curvature_change
                
                if max(abs(new_dy), abs(new_dx)) == 2:
                    mid_y, mid_x = y + new_dy//2, x + new_dx//2
                    path.append((mid_y, mid_x))
                    temp_path_visited.add((mid_y, mid_x))
                
                y, x = best_pixel
                prev_dy, prev_dx = new_dy, new_dx
            else:
                break
        
        processed_starts.add((node_id, initial_y, initial_x))
    
    # Draw nodes
    for node_id, data in nodes.items():
        x, y = data['pos']
        if data['type'] == 0:
            color = (255, 0, 0)  # Intersection: Red
        elif data['type'] == 1:
            color = (255, 0, 255)  # Corner: Magenta
        elif data['type'] == 2:
            color = (0, 255, 255)  # Endpoint: Yellow
        elif data['type'] == 3:
            color = (0, 165, 255)  # Curvature split: Orange
        else:
            color = (128, 128, 128)  # Other: Gray
        
        radius = 5 if data['type'] != 3 else 3
        cv2.circle(marked_img, (x, y), radius, color, -1)
    
    return nodes, edges, marked_img, debug_info


def integrate_isolated_networks(nodes, edges, distance_threshold=30):
    """
    Integrate isolated network components by connecting them to the main component
    
    Parameters:
    - nodes: Dictionary of node data
    - edges: Set of edges (tuples of node IDs)
    - distance_threshold: Maximum distance in pixels to connect components
    
    Returns:
    - updated_edges: Set of edges including new bridging edges
    - integration_info: Dictionary containing integration statistics
    """
    # Build graph using NetworkX
    G = nx.Graph()
    G.add_nodes_from(nodes.keys())
    G.add_edges_from(edges)
    
    # Find connected components
    connected_components = list(nx.connected_components(G))
    connected_components.sort(key=len, reverse=True)
    
    if len(connected_components) == 1:
        return edges, {
            'num_components_before': 1,
            'num_components_after': 1,
            'new_edges_added': 0,
            'is_fully_integrated': True
        }
    
    main_component = connected_components[0]
    new_edges = []
    
    # Connect each isolated component to the main component
    for comp in connected_components[1:]:
        comp_nodes = list(comp)
        
        min_distance = float('inf')
        best_pair = None
        
        # Find closest node pair between this component and main component
        for comp_node_id in comp_nodes:
            comp_pos = nodes[comp_node_id]['pos']
            comp_x, comp_y = comp_pos
            
            for main_node_id in main_component:
                main_pos = nodes[main_node_id]['pos']
                main_x, main_y = main_pos
                
                dist = np.sqrt((comp_x - main_x)**2 + (comp_y - main_y)**2)
                
                if dist < min_distance:
                    min_distance = dist
                    best_pair = (comp_node_id, main_node_id, dist)
        
        # Add connection if within threshold
        if best_pair and min_distance <= distance_threshold:
            n1, n2 = min(best_pair[0], best_pair[1]), max(best_pair[0], best_pair[1])
            edge_key = (n1, n2)
            
            if edge_key not in edges:
                new_edges.append(edge_key)
                
                # Update node adjacency lists
                pixel_dist = int(round(best_pair[2]))
                
                if best_pair[1] not in [adj[0] for adj in nodes[best_pair[0]]['adj']]:
                    nodes[best_pair[0]]['adj'].append((best_pair[1], pixel_dist))
                
                if best_pair[0] not in [adj[0] for adj in nodes[best_pair[1]]['adj']]:
                    nodes[best_pair[1]]['adj'].append((best_pair[0], pixel_dist))
    
    # Combine original and new edges
    updated_edges = edges.copy()
    updated_edges.update(new_edges)
    
    # Verify integration
    G_new = nx.Graph()
    G_new.add_nodes_from(nodes.keys())
    G_new.add_edges_from(updated_edges)
    num_components_after = nx.number_connected_components(G_new)
    
    integration_info = {
        'num_components_before': len(connected_components),
        'num_components_after': num_components_after,
        'new_edges_added': len(new_edges),
        'is_fully_integrated': num_components_after == 1,
        'component_sizes_before': [len(comp) for comp in connected_components]
    }
    
    return updated_edges, integration_info


def create_csv_data(nodes, edges, image_height, meters_per_pixel=None):
    """Create CSV data (output as bidirectional edges)"""
    type_labels = {
        0: 'Intersection',
        1: 'Corner',
        2: 'Endpoint',
        3: 'Intermediate (Curvature Split)'
    }
    
    # Node CSV
    node_data = []
    for node_id, data in nodes.items():
        x_pixel, y_pixel = data['pos']
        node_type = data['type']
        
        # Convert to Scratch coordinate system (origin at center, y-axis upward)
        x_scratch = int(round(x_pixel - 240))  # For 480 width
        y_scratch = int(round(image_height / 2 - y_pixel))
        
        node_data.append([
            node_id,
            x_scratch,
            y_scratch,
            node_type,
            type_labels.get(node_type, 'Unknown')
        ])
    
    # Edge CSV (output as bidirectional)
    edge_data = []
    edge_id = 1
    
    # Generate bidirectional edges from edges set (undirected edges)
    for n1, n2 in edges:
        # Get edge length (from n1's adj list)
        length = None
        for neighbor_id, edge_length in nodes[n1]['adj']:
            if neighbor_id == n2:
                length = edge_length
                break
        
        if length is None:
            # Also try from n2's adj list
            for neighbor_id, edge_length in nodes[n2]['adj']:
                if neighbor_id == n1:
                    length = edge_length
                    break
        
        if length is None:
            continue
        
        # Output as 2 rows for bidirectional edge
        if meters_per_pixel is not None:
            distance_meters = length * meters_per_pixel
            
            # n1 -> n2
            edge_data.append([edge_id, n1, n2, length, f"{distance_meters:.2f}"])
            edge_id += 1
            
            # n2 -> n1 (reverse direction)
            edge_data.append([edge_id, n2, n1, length, f"{distance_meters:.2f}"])
            edge_id += 1
        else:
            # n1 -> n2
            edge_data.append([edge_id, n1, n2, length])
            edge_id += 1
            
            # n2 -> n1 (reverse direction)
            edge_data.append([edge_id, n2, n1, length])
            edge_id += 1
    
    return node_data, edge_data


def create_csv_file(data, header):
    """Create CSV string"""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(header)
    writer.writerows(data)
    return output.getvalue()


# --- Main processing ---

if uploaded_file is not None:
    # Load image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    st.success("✅ Image uploaded successfully")
    
    # Display distance scale calculation
    if enable_distance_scale:
        m_per_px_x, m_per_px_y, m_per_px_avg = calculate_distance_scale(
            north_latitude, 
            south_latitude,
            west_longitude, 
            east_longitude, 
            image_width_px,
            image_height_px
        )
        
        center_lat = (north_latitude + south_latitude) / 2
        
        st.info(f"📏 **Distance Scale Calculation Results** (Center latitude: {center_lat:.6f}°)\n\n"
                f"- Horizontal: 1px = {m_per_px_x:.2f} m (Longitude diff {abs(east_longitude - west_longitude):.6f}°)\n"
                f"- Vertical: 1px = {m_per_px_y:.2f} m (Latitude diff {abs(north_latitude - south_latitude):.6f}°)\n"
                f"- Average: 1px = {m_per_px_avg:.2f} m")
    
    # Processing execution button
    if st.button("🚀 Generate Graph Data", type="primary"):
        with st.spinner("Processing..."):
            progress_bar = st.progress(0)
            
            # Step 1: Resize
            if resize_enabled:
                st.info("Step 1/4: Resizing image...")
                img, orig_h, orig_w = resize_image(img, 480, 360)
                current_height = 360
                progress_bar.progress(20)
            else:
                current_height = img.shape[0]
            
            # Step 2: Skeletonization
            st.info("Step 2/4: Skeletonizing...")
            skeleton_data, skeleton_visual = high_quality_skeletonization(img)
            progress_bar.progress(50)
            
            # Step 3: Graph construction
            st.info("Step 3/4: Building graph...")
            nodes_data, edges_set, marked_img, debug_info = detect_and_build_graph(
                skeleton_data,
                curvature_threshold,
                max_jump_distance,
                min_intersection_transitions,
                min_node_area,
                debug=debug_mode
            )
            progress_bar.progress(75)
            
            # Display debug info
            if debug_mode and debug_info:
                with st.expander("🐛 Debug Information", expanded=True):
                    col_d1, col_d2, col_d3 = st.columns(3)
                    with col_d1:
                        st.metric("Skeleton Pixels", debug_info['total_skeleton_pixels'])
                        st.metric("Intersections", debug_info['intersections_found'])
                    with col_d2:
                        st.metric("Corners", debug_info['corners_found'])
                        st.metric("Endpoints", debug_info['endpoints_found'])
                    with col_d3:
                        st.metric("Feature Pixels", debug_info['feature_pixels_before_clustering'])
                        st.metric("Regions Found", debug_info['regions_found'])
                    st.metric("Regions Kept (min_area filter)", debug_info['regions_kept'])
            
            if nodes_data is None or edges_set is None:
                st.error("❌ Graph detection failed. Please adjust parameters.")
                if debug_mode and debug_info:
                    st.info("💡 **Suggestions:**")
                    if debug_info['total_skeleton_pixels'] == 0:
                        st.write("- No skeleton pixels found. The image might be too light or preprocessing failed.")
                    elif debug_info['feature_pixels_before_clustering'] == 0:
                        st.write("- No feature points detected. Try lowering 'Intersection detection threshold' to 2.")
                    elif debug_info['regions_kept'] == 0:
                        st.write("- All regions filtered out. Try lowering 'Minimum node area' to 1.")
            else:
                # Step 4: Network integration (optional)
                integration_info = None
                if enable_integration:
                    st.info("Step 4/4: Integrating isolated networks...")
                    edges_set, integration_info = integrate_isolated_networks(
                        nodes_data, 
                        edges_set, 
                        integration_threshold
                    )
                
                progress_bar.progress(100)
                
                st.success(f"✅ Processing complete! Nodes: {len(nodes_data)}, Edges: {len(edges_set)}")
                
                # Display integration results
                if integration_info:
                    if integration_info['is_fully_integrated']:
                        st.success(f"🔗 **Network fully integrated!** "
                                 f"({integration_info['num_components_before']} → 1 component, "
                                 f"added {integration_info['new_edges_added']} connections)")
                    else:
                        st.warning(f"⚠️ **Partial integration**: "
                                 f"{integration_info['num_components_before']} → "
                                 f"{integration_info['num_components_after']} components. "
                                 f"Try increasing the integration distance threshold.")
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("Original Image")
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                with col2:
                    st.subheader("Skeleton Image")
                    st.image(skeleton_visual, use_container_width=True)
                
                with col3:
                    st.subheader("Graph Image")
                    st.image(cv2.cvtColor(marked_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # Generate CSV data
                if enable_distance_scale:
                    node_data, edge_data = create_csv_data(
                        nodes_data, edges_set, current_height, m_per_px_avg
                    )
                else:
                    node_data, edge_data = create_csv_data(
                        nodes_data, edges_set, current_height
                    )
                
                # Download buttons
                st.subheader("📥 Data Download")
                
                col_dl1, col_dl2, col_dl3 = st.columns(3)
                
                with col_dl1:
                    node_csv = create_csv_file(
                        node_data,
                        ['node_id', 'x_scratch', 'y_scratch', 'type_code', 'type_label']
                    )
                    st.download_button(
                        label="Download Node CSV",
                        data=node_csv,
                        file_name="nodes.csv",
                        mime="text/csv"
                    )
                
                with col_dl2:
                    if enable_distance_scale:
                        edge_header = ['edge_id', 'from_node_id', 'to_node_id', 'pixel_length', 'distance_meters']
                    else:
                        edge_header = ['edge_id', 'from_node_id', 'to_node_id', 'pixel_length']
                    
                    edge_csv = create_csv_file(edge_data, edge_header)
                    st.download_button(
                        label="Download Edge CSV",
                        data=edge_csv,
                        file_name="edges.csv",
                        mime="text/csv"
                    )
                
                with col_dl3:
                    # Download graph image
                    is_success, buffer = cv2.imencode(".png", marked_img)
                    if is_success:
                        st.download_button(
                            label="Download Graph Image",
                            data=buffer.tobytes(),
                            file_name="graph_marked.png",
                            mime="image/png"
                        )
                
                # Data preview
                with st.expander("📊 Node Data Preview"):
                    st.write(f"Total nodes: {len(node_data)}")
                    df_nodes = pd.DataFrame(
                        node_data,
                        columns=['node_id', 'x_scratch', 'y_scratch', 'type_code', 'type_label']
                    )
                    st.dataframe(df_nodes.head(10))
                
                with st.expander("🔗 Edge Data Preview"):
                    st.write(f"Total edges: {len(edge_data)}")
                    if enable_distance_scale:
                        df_edges = pd.DataFrame(
                            edge_data,
                            columns=['edge_id', 'from_node_id', 'to_node_id', 'pixel_length', 'distance_meters']
                        )
                    else:
                        df_edges = pd.DataFrame(
                            edge_data,
                            columns=['edge_id', 'from_node_id', 'to_node_id', 'pixel_length']
                        )
                    st.dataframe(df_edges.head(10))
                    
                    # Display distance statistics
                    if enable_distance_scale:
                        st.markdown("**Distance Statistics**")
                        total_distance = sum([float(row[4]) for row in edge_data])
                        avg_distance = total_distance / len(edge_data) if edge_data else 0
                        st.write(f"- Total distance: {total_distance:.2f} m ({total_distance/1000:.2f} km)")
                        st.write(f"- Average edge length: {avg_distance:.2f} m")

else:
    st.info("👆 Please select an image from the file uploader on the left")
    
    # Usage instructions
    with st.expander("📖 How to Use"):
        st.markdown("""
        ### How to Use
        
        1. **Upload Image**: Select an image file from the sidebar
        2. **Distance Scale Settings** (Optional): Enable real distance calculation and enter latitude/longitude range
        3. **Network Integration** (Optional): Enable to automatically connect isolated network components
        4. **Adjust Parameters**: Adjust various parameters in the sidebar
        5. **Enable Debug Mode**: Check debug mode to see detailed processing information
        6. **Generate**: Click the "Generate Graph Data" button
        7. **Review Results**: Check the generated graph and data
        8. **Download**: Download CSV files and images
        
        ### Parameter Descriptions
        
        #### Distance Scale Settings
        - **Enable real distance calculation**: Convert pixel length to real distance (meters)
        - **North/South Latitude**: Top and bottom latitude of the image
        - **West/East Longitude**: Left and right longitude of the image
        - **Image Size**: Width and height of the image after resizing (pixels)
        
        #### Network Integration
        - **Integrate isolated networks**: Automatically connect disconnected network components
        - **Integration distance threshold**: Maximum distance (pixels) to bridge isolated components
        
        #### Image Processing
        - **Image Resize**: Resize to 480x360 for improved processing speed
        - **Curvature split threshold**: Larger values make it easier to recognize as straight lines (1-30)
        - **Max jump distance**: Noise tolerance (2 recommended normally)
        - **Intersection detection threshold**: Sensitivity of intersection detection (2 will also detect corners)
        - **Minimum node area**: Remove small noise
        
        #### Debug Mode
        - **Enable debug mode**: Shows detailed statistics about detection process
        - Helps identify why nodes aren't being detected
        - Displays counts of intersections, corners, endpoints, and filtering results
        
        ### Node Types
        
        The system automatically detects four types of nodes:
        - **Intersection** (Red): Points where 3+ paths meet
        - **Corner** (Magenta): Points where exactly 2 paths meet at an angle
        - **Endpoint** (Yellow): Terminal points where paths end
        - **Curvature Split** (Orange): Points added to split curved paths
        
        ### Troubleshooting
        
        If no nodes are detected:
        1. Enable debug mode to see what's happening
        2. Check if skeleton pixels are found (if 0, image might be too light)
        3. Lower "Intersection detection threshold" to 2 to detect corners
        4. Lower "Minimum node area" to 1 to keep small features
        5. Try adjusting the adaptive threshold parameters in preprocessing
        """)
    
    # Color legend
    with st.expander("🎨 Node Color Meanings"):
        col_legend1, col_legend2, col_legend3, col_legend4 = st.columns(4)
        
        with col_legend1:
            st.markdown("🔴 **Red**: Intersection")
        with col_legend2:
            st.markdown("🟣 **Magenta**: Corner")
        with col_legend3:
            st.markdown("🟡 **Yellow**: Endpoint")
        with col_legend4:
            st.markdown("🟠 **Orange**: Curvature split")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
