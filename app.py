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

# Page Configuration
st.set_page_config(
    page_title="Road Network Graph Extraction System",
    page_icon="🗺️",
    layout="wide"
)

# Title and Description
st.title("🗺️ Road Network Graph Extraction from Raster Images")
st.markdown("""
**Abstract**: This application implements an automated pipeline for extracting graph representations 
from raster road network images through skeletonization and topological analysis.
""")

# Sidebar Parameters
st.sidebar.header("⚙️ Configuration Parameters")

# Geographic Calibration
st.sidebar.subheader("📍 Geographic Calibration")
enable_geographic = st.sidebar.checkbox("Enable Geographic Calibration", value=False)

if enable_geographic:
    st.sidebar.markdown("**Bounding Box Coordinates (WGS84)**")
    
    col_lat1, col_lat2 = st.sidebar.columns(2)
    with col_lat1:
        north_lat = st.number_input("North Latitude", value=35.1, format="%.6f", step=0.000001)
    with col_lat2:
        south_lat = st.number_input("South Latitude", value=35.0, format="%.6f", step=0.000001)
    
    col_lon1, col_lon2 = st.sidebar.columns(2)
    with col_lon1:
        west_lon = st.number_input("West Longitude", value=135.0, format="%.6f", step=0.000001)
    with col_lon2:
        east_lon = st.number_input("East Longitude", value=135.1, format="%.6f", step=0.000001)
    
    # Image dimensions
    st.sidebar.markdown("**Image Dimensions (pixels)**")
    col_size1, col_size2 = st.sidebar.columns(2)
    with col_size1:
        img_width = st.number_input("Width", value=480, min_value=1)
    with col_size2:
        img_height = st.number_input("Height", value=360, min_value=1)

# Image Processing Parameters
st.sidebar.subheader("Image Processing")
resize_enabled = st.sidebar.checkbox("Resize to 480×360", value=True)

# Graph Construction Parameters
st.sidebar.subheader("Graph Construction")
curvature_threshold = st.sidebar.slider("Curvature Threshold", 1.0, 20.0, 10.0, 0.5,
                                        help="Higher values result in fewer intermediate nodes")
max_jump_distance = st.sidebar.slider("Max Jump Distance", 1, 5, 2,
                                      help="Maximum pixel distance for edge continuity")
min_intersection_transitions = st.sidebar.slider("Intersection Detection Threshold", 2, 5, 3,
                                                 help="Minimum transitions for intersection detection")
min_node_area = st.sidebar.slider("Minimum Node Area", 1, 10, 1,
                                  help="Minimum pixel area for valid nodes")

# File Upload
uploaded_file = st.file_uploader("Select Road Network Image", type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'])

# --- Core Functions ---

def calculate_geographic_scale(north_lat, south_lat, west_lon, east_lon, width_px, height_px):
    """
    Calculate the geographic scale factor for distance measurements.
    
    Parameters:
    -----------
    north_lat, south_lat : float
        Northern and southern latitude bounds
    west_lon, east_lon : float
        Western and eastern longitude bounds
    width_px, height_px : int
        Image dimensions in pixels
    
    Returns:
    --------
    tuple : (meters_per_pixel_x, meters_per_pixel_y, meters_per_pixel_avg)
    """
    EARTH_RADIUS = 6371000  # meters
    
    center_lat = (north_lat + south_lat) / 2
    center_lat_rad = math.radians(center_lat)
    
    # Longitudinal distance
    lon_diff = abs(east_lon - west_lon)
    lon_diff_rad = math.radians(lon_diff)
    distance_x = EARTH_RADIUS * lon_diff_rad * math.cos(center_lat_rad)
    meters_per_pixel_x = distance_x / width_px
    
    # Latitudinal distance
    lat_diff = abs(north_lat - south_lat)
    lat_diff_rad = math.radians(lat_diff)
    distance_y = EARTH_RADIUS * lat_diff_rad
    meters_per_pixel_y = distance_y / height_px
    
    meters_per_pixel_avg = (meters_per_pixel_x + meters_per_pixel_y) / 2
    
    return meters_per_pixel_x, meters_per_pixel_y, meters_per_pixel_avg


def resize_image(img, target_width=480, target_height=360):
    """Resize image to target dimensions."""
    original_height, original_width = img.shape[:2]
    resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized, original_height, original_width


def refine_skeleton_branches(skeleton):
    """Remove spurious branches from skeleton."""
    H, W = skeleton.shape
    refined = skeleton.copy()
    
    neighbors_8 = [(-1, -1), (-1, 0), (-1, 1), (0, 1), 
                   (1, 1), (1, 0), (1, -1), (0, -1)]
    
    # Find endpoints
    endpoints = []
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if skeleton[y, x] == 1:
                neighbors = [skeleton[y + dy, x + dx] for dy, dx in neighbors_8]
                if sum(neighbors) == 1:
                    endpoints.append((y, x))
    
    # Remove short branches
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


def morphological_skeletonization(img):
    """
    Apply morphological skeletonization to extract road centerlines.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image
    
    Returns:
    --------
    tuple : (skeleton_binary, skeleton_visual)
    """
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Denoise
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Adaptive thresholding for robust binarization
    binary = cv2.adaptiveThreshold(
        denoised, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        blockSize=11, 
        C=2
    )
    
    # Morphological cleaning
    kernel_small = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
    
    kernel_dilate = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(cleaned, kernel_dilate, iterations=1)
    
    # Skeletonization
    binary_bool = (dilated > 128).astype(bool)
    skeleton_bool = skeletonize(binary_bool)
    skeleton = skeleton_bool.astype(np.uint8)
    
    # Component filtering
    labeled_skeleton = label(skeleton, connectivity=2)
    regions = regionprops(labeled_skeleton)
    
    min_component_size = 5
    filtered_skeleton = np.zeros_like(skeleton)
    for region in regions:
        if region.area >= min_component_size:
            for coord in region.coords:
                filtered_skeleton[coord[0], coord[1]] = 1
    
    # Refine skeleton
    filtered_skeleton = refine_skeleton_branches(filtered_skeleton)
    
    processed_img = (filtered_skeleton * 255).astype(np.uint8)
    
    return filtered_skeleton, processed_img


def extract_graph_topology(binary_img, curvature_threshold, max_jump, min_transitions, min_area):
    """
    Extract topological graph structure from skeleton image.
    
    Parameters:
    -----------
    binary_img : numpy.ndarray
        Binary skeleton image
    curvature_threshold : float
        Threshold for curvature-based node splitting
    max_jump : int
        Maximum jump distance for edge tracing
    min_transitions : int
        Minimum transitions for intersection detection
    min_area : int
        Minimum node area
    
    Returns:
    --------
    tuple : (nodes_dict, edges_set, annotated_image)
    """
    H, W = binary_img.shape
    
    feature_map = np.zeros_like(binary_img)
    neighbors_coord = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    feature_pixels = {}
    
    # Feature detection (intersections, endpoints, high-curvature points)
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
                elif transitions == 1:
                    is_feature = True
                    node_type = 2  # Endpoint
                elif transitions == 2:
                    white_indices = [i for i, val in enumerate(neighbors) if val]
                    if len(white_indices) == 2:
                        idx1, idx2 = white_indices
                        distance = min(abs(idx1 - idx2), 8 - abs(idx1 - idx2))
                        if distance == 2:
                            is_feature = True
                            node_type = 1  # High-curvature point
                
                if is_feature:
                    feature_map[y, x] = 1
                    feature_pixels[(y, x)] = node_type
    
    if feature_map.sum() == 0:
        return None, None, None
    
    # Node clustering and consolidation
    labeled_img = label(feature_map, connectivity=2)
    regions = regionprops(labeled_img)
    
    nodes = {}
    coord_to_node_id = np.full((H, W), -1, dtype=int)
    node_id_counter = 1
    
    for region in regions:
        if region.area < min_area:
            continue
        
        node_id = node_id_counter
        center_y, center_x = region.centroid
        
        # Determine node type from cluster
        cluster_types = [feature_pixels[(py, px)] for py, px in region.coords 
                         if (py, px) in feature_pixels]
        if cluster_types:
            most_common_type = collections.Counter(cluster_types).most_common(1)[0][0]
        else:
            continue
        
        # Node consolidation with dilation
        dilation_radius = max_jump
        pixels_to_map = set()
        
        for y, x in region.coords:
            pixels_to_map.add((y, x))
            
        for y_orig, x_orig in region.coords:
            for dy in range(-dilation_radius, dilation_radius + 1):
                for dx in range(-dilation_radius, dilation_radius + 1):
                    ny, nx = y_orig + dy, x_orig + dx
                    if (0 <= ny < H and 0 <= nx < W and binary_img[ny, nx] == 1):
                        pixels_to_map.add((ny, nx))
        
        mapped_coords = []
        for y, x in pixels_to_map:
            if coord_to_node_id[y, x] == -1:
                coord_to_node_id[y, x] = node_id
                mapped_coords.append((y, x))
        
        if not mapped_coords:
            continue
        
        nodes[node_id] = {
            'pos': (int(center_x), int(center_y)),
            'type': most_common_type,
            'adj': [],
            'coords': mapped_coords
        }
        
        node_id_counter += 1
    
    if len(nodes) == 0:
        return None, None, None
    
    # Edge tracing
    marked_img = cv2.cvtColor(binary_img * 255, cv2.COLOR_GRAY2BGR)
    edges = set()
    edge_visited_map = np.full((H, W), -1, dtype=int)
    edge_id_counter = 0
    
    # Initialize edge starting points
    start_pixels = []
    for node_id, node_data in nodes.items():
        for start_y, start_x in node_data['coords']:
            for dy, dx in neighbors_coord:
                neighbor_y, neighbor_x = start_y + dy, start_x + dx
                if (0 <= neighbor_y < H and 0 <= neighbor_x < W and 
                    binary_img[neighbor_y, neighbor_x] == 1 and 
                    coord_to_node_id[neighbor_y, neighbor_x] == -1):
                    start_pixels.append((node_id, start_y, start_x, neighbor_y, neighbor_x))
    
    processed_starts = set()
    
    # Trace edges
    for node_id, start_y, start_x, initial_y, initial_x in start_pixels:
        start_key = (node_id, initial_y, initial_x)
        if start_key in processed_starts or edge_visited_map[initial_y, initial_x] != -1:
            continue
        
        path = []
        temp_path_visited = set()
        y, x = initial_y, initial_x
        prev_dy, prev_dx = initial_y - start_y, initial_x - start_x
        current_curvature = 0.0
        current_start_node_id = node_id
        
        while True:
            # Check termination conditions
            end_node_id_check = coord_to_node_id[y, x]
            is_end_node = (end_node_id_check != -1 and 
                          end_node_id_check != current_start_node_id)
            is_split_point = (current_curvature >= curvature_threshold and 
                            end_node_id_check == -1)
            
            if is_end_node or is_split_point:
                if is_end_node:
                    target_node_id = end_node_id_check
                else:
                    target_node_id = node_id_counter
                    nodes[target_node_id] = {
                        'pos': (x, y),
                        'type': 3,  # Intermediate node
                        'adj': [],
                        'coords': [(y, x)]
                    }
                    coord_to_node_id[y, x] = target_node_id
                    node_id_counter += 1
                
                n1, n2 = min(current_start_node_id, target_node_id), max(current_start_node_id, target_node_id)
                edge_key = (n1, n2)
                
                if edge_key not in edges:
                    edges.add(edge_key)
                    length = len(path)
                    
                    # Update adjacency lists
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
            
            # Find next pixel
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
                    
                    if ((next_y, next_x) in temp_path_visited or 
                        edge_visited_map[next_y, next_x] != -1):
                        continue
                    
                    if binary_img[next_y, next_x] == 1:
                        if (coord_to_node_id[next_y, next_x] != -1 and 
                            coord_to_node_id[next_y, next_x] != current_start_node_id):
                            best_pixel = (next_y, next_x)
                            best_vector = (dy_search, dx_search)
                            best_score = 10
                            break
                        
                        is_adjacent = max(abs(dy_search), abs(dx_search)) == 1
                        is_jump = max(abs(dy_search), abs(dx_search)) == 2
                        
                        if is_adjacent:
                            score = prev_dy * dy_search + prev_dx * dx_search
                            if score > best_score:
                                best_score = score
                                best_pixel = (next_y, next_x)
                                best_vector = (dy_search, dx_search)
                        
                        elif is_jump:
                            mid_y, mid_x = y + dy_search//2, x + dx_search//2
                            if binary_img[mid_y, mid_x] == 0:
                                score = prev_dy * dy_search + prev_dx * dx_search - 3
                                if score > best_score:
                                    best_score = score
                                    best_pixel = (next_y, next_x)
                                    best_vector = (dy_search, dx_search)
                
                if best_score == 10:
                    break
            
            if best_pixel:
                new_dy, new_dx = best_vector
                
                if (coord_to_node_id[best_pixel[0], best_pixel[1]] != -1 and 
                    coord_to_node_id[best_pixel[0], best_pixel[1]] != current_start_node_id):
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
    
    # Visualize nodes
    for node_id, data in nodes.items():
        x, y = data['pos']
        if data['type'] == 0:
            color = (255, 0, 0)  # Red: Intersection
        elif data['type'] == 1:
            color = (0, 0, 255)  # Blue: High-curvature
        elif data['type'] == 2:
            color = (0, 255, 255)  # Yellow: Endpoint
        elif data['type'] == 3:
            color = (0, 165, 255)  # Orange: Intermediate
        
        radius = 5 if data['type'] != 3 else 3
        cv2.circle(marked_img, (x, y), radius, color, -1)
    
    return nodes, edges, marked_img


def generate_graph_data(nodes, edges, image_height, meters_per_pixel=None):
    """Generate structured graph data for export."""
    type_labels = {
        0: 'Intersection',
        1: 'High_Curvature',
        2: 'Endpoint',
        3: 'Intermediate'
    }
    
    # Node data
    node_data = []
    for node_id, data in nodes.items():
        x_pixel, y_pixel = data['pos']
        node_type = data['type']
        
        # Convert to Cartesian coordinates (origin at center)
        x_cartesian = int(round(x_pixel - 240))
        y_cartesian = int(round(image_height / 2 - y_pixel))
        
        node_data.append([
            node_id,
            x_cartesian,
            y_cartesian,
            node_type,
            type_labels.get(node_type, 'Unknown')
        ])
    
    # Edge data
    edge_data = []
    edge_id_counter = 1
    unique_edges = set()
    
    for node_id, data in nodes.items():
        for neighbor_id, length in data['adj']:
            n1, n2 = min(node_id, neighbor_id), max(node_id, neighbor_id)
            edge_key = (n1, n2)
            
            if edge_key not in unique_edges:
                unique_edges.add(edge_key)
                
                if meters_per_pixel is not None:
                    distance_meters = length * meters_per_pixel
                    edge_data.append([edge_id_counter, n1, n2, length, f"{distance_meters:.2f}"])
                else:
                    edge_data.append([edge_id_counter, n1, n2, length])
                
                edge_id_counter += 1
    
    return node_data, edge_data


def create_csv_file(data, header):
    """Create CSV string from data."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(header)
    writer.writerows(data)
    return output.getvalue()


# --- Main Application ---

if uploaded_file is not None:
    # Load image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    st.success("✅ Image loaded successfully")
    
    # Geographic calibration display
    if enable_geographic:
        m_per_px_x, m_per_px_y, m_per_px_avg = calculate_geographic_scale(
            north_lat, south_lat,
            west_lon, east_lon,
            img_width, img_height
        )
        
        center_lat = (north_lat + south_lat) / 2
        
        st.info(f"""
        📍 **Geographic Scale Calibration** (Center: {center_lat:.6f}°)
        - X-axis: 1 px = {m_per_px_x:.2f} m
        - Y-axis: 1 px = {m_per_px_y:.2f} m
        - Average: 1 px = {m_per_px_avg:.2f} m
        """)
    
    # Process button
    if st.button("🔬 Extract Graph Structure", type="primary"):
        with st.spinner("Processing..."):
            progress_bar = st.progress(0)
            
            # Step 1: Resize
            if resize_enabled:
                st.info("Phase 1/3: Image preprocessing...")
                img, orig_h, orig_w = resize_image(img, 480, 360)
                current_height = 360
                progress_bar.progress(25)
            else:
                current_height = img.shape[0]
            
            # Step 2: Skeletonization
            st.info("Phase 2/3: Morphological skeletonization...")
            skeleton_data, skeleton_visual = morphological_skeletonization(img)
            progress_bar.progress(60)
            
            # Step 3: Graph extraction
            st.info("Phase 3/3: Topological graph extraction...")
            nodes_data, edges_set, marked_img = extract_graph_topology(
                skeleton_data,
                curvature_threshold,
                max_jump_distance,
                min_intersection_transitions,
                min_node_area
            )
            progress_bar.progress(100)
            
            if nodes_data is None or edges_set is None:
                st.error("❌ Graph extraction failed. Please adjust parameters.")
            else:
                st.success(f"✅ Extraction complete! Nodes: {len(nodes_data)}, Edges: {len(edges_set)}")
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("Original Image")
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                with col2:
                    st.subheader("Skeleton Image")
                    st.image(skeleton_visual, use_container_width=True)
                
                with col3:
                    st.subheader("Graph Structure")
                    st.image(cv2.cvtColor(marked_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # Generate CSV data
                if enable_geographic:
                    node_data, edge_data = generate_graph_data(
                        nodes_data, edges_set, current_height, m_per_px_avg
                    )
                else:
                    node_data, edge_data = generate_graph_data(
                        nodes_data, edges_set, current_height
                    )
                
                # Export section
                st.subheader("📊 Export Graph Data")
                
                col_dl1, col_dl2, col_dl3 = st.columns(3)
                
                with col_dl1:
                    node_csv = create_csv_file(
                        node_data,
                        ['node_id', 'x_coord', 'y_coord', 'type_code', 'type_label']
                    )
                    st.download_button(
                        label="Download Nodes CSV",
                        data=node_csv,
                        file_name="graph_nodes.csv",
                        mime="text/csv"
                    )
                
                with col_dl2:
                    if enable_geographic:
                        edge_header = ['edge_id', 'source_node', 'target_node', 'pixel_length', 'distance_meters']
                    else:
                        edge_header = ['edge_id', 'source_node', 'target_node', 'pixel_length']
                    
                    edge_csv = create_csv_file(edge_data, edge_header)
                    st.download_button(
                        label="Download Edges CSV",
                        data=edge_csv,
                        file_name="graph_edges.csv",
                        mime="text/csv"
                    )
                
                with col_dl3:
                    is_success, buffer = cv2.imencode(".png", marked_img)
                    if is_success:
                        st.download_button(
                            label="Download Graph Visualization",
                            data=buffer.tobytes(),
                            file_name="graph_visualization.png",
                            mime="image/png"
                        )
                
                # Data preview with statistics
                with st.expander("📈 Graph Statistics and Preview"):
                    import pandas as pd
                    
                    # Node statistics
                    st.markdown("### Node Statistics")
                    df_nodes = pd.DataFrame(
                        node_data,
                        columns=['node_id', 'x_coord', 'y_coord', 'type_code', 'type_label']
                    )
                    
                    node_type_counts = df_nodes['type_label'].value_counts()
                    col_stat1, col_stat2 = st.columns(2)
                    
                    with col_stat1:
                        st.metric("Total Nodes", len(node_data))
                        for node_type, count in node_type_counts.items():
                            st.metric(node_type, count)
                    
                    with col_stat2:
                        st.dataframe(df_nodes.head(10))
                    
                    # Edge statistics
                    st.markdown("### Edge Statistics")
                    if enable_geographic:
                        df_edges = pd.DataFrame(
                            edge_data,
                            columns=['edge_id', 'source_node', 'target_node', 'pixel_length', 'distance_meters']
                        )
                        
                        total_distance = sum([float(row[4]) for row in edge_data])
                        avg_distance = total_distance / len(edge_data) if edge_data else 0
                        
                        col_stat3, col_stat4 = st.columns(2)
                        with col_stat3:
                            st.metric("Total Edges", len(edge_data))
                            st.metric("Total Network Length", f"{total_distance:.2f} m")
                            st.metric("Average Edge Length", f"{avg_distance:.2f} m")
                        
                        with col_stat4:
                            st.dataframe(df_edges.head(10))
                    else:
                        df_edges = pd.DataFrame(
                            edge_data,
                            columns=['edge_id', 'source_node', 'target_node', 'pixel_length']
                        )
                        st.metric("Total Edges", len(edge_data))
                        st.dataframe(df_edges.head(10))

else:
    st.info("Please upload a road network image to begin analysis")
    
    # Documentation
    with st.expander("📚 Documentation"):
        st.markdown("""
        ## Methodology
        
        This system implements a three-phase pipeline for road network extraction:
        
        ### Phase 1: Image Preprocessing
        - **Adaptive thresholding**: Robust binarization for varying illumination
        - **Morphological operations**: Noise reduction and gap filling
        - **Optional resizing**: Computational optimization
        
        ### Phase 2: Morphological Skeletonization
        - **Medial axis transform**: Extract road centerlines
        - **Branch pruning**: Remove spurious branches < 5 pixels
        - **Component filtering**: Remove isolated noise components
        
        ### Phase 3: Topological Graph Extraction
        - **Feature detection**: Identify intersections, endpoints, and high-curvature points
        - **Node clustering**: Consolidate nearby feature points
        - **Edge tracing**: Connect nodes following skeleton paths
        - **Curvature-based splitting**: Create intermediate nodes at high-curvature locations
        
        ## Parameters
        
        - **Curvature Threshold**: Controls edge segmentation at curves (default: 10.0)
        - **Max Jump Distance**: Gap-bridging capability in pixels (default: 2)
        - **Intersection Detection**: Sensitivity for junction detection (default: 3)
        - **Minimum Node Area**: Filter threshold for valid nodes (default: 1)
        
        ## Output Format
        
        The system generates:
        1. **Node CSV**: Node ID, coordinates, type classification
        2. **Edge CSV**: Edge ID, source/target nodes, length metrics
        3. **Visualization**: Annotated graph structure overlay
        
        ## Node Types
        - **Intersection** (Red): Road junctions with degree ≥ 3
        - **High Curvature** (Blue): Sharp turns in road geometry
        - **Endpoint** (Yellow): Terminal nodes with degree = 1
        - **Intermediate** (Orange): Curvature-based segmentation points
        
        ## Geographic Calibration
        
        When enabled, the system calculates real-world distances using:
        - WGS84 coordinate system
        - Haversine formula for geographic distance
        - Latitude-adjusted scale factors
        
        ## Citation
        
        If you use this tool in your research, please cite:
```
        @software{road_network_extraction_2024,
          title = {Road Network Graph Extraction System},
          author = {[Your Name]},
          year = {2024},
          url = {https://github.com/yourusername/road-network-extraction}
        }
```
        """)

# Footer
st.markdown("---")
st.markdown("**Road Network Graph Extraction System** | Version 1.0 | Academic Research Tool")
