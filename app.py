import streamlit as st
import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
import collections
import csv
import io
from PIL import Image
import math
import networkx as nx
import pandas as pd
from sklearn.decomposition import PCA

# --- Streamlit page configuration ---
st.set_page_config(
    page_title="Image Graph Generation (Curvature-integrated)",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Generate Graph Data from Image (Curvature-integrated)")
st.markdown("Upload a line-based map image to extract intersections, curves, and endpoints, and export CSV graph data.")

# Sidebar settings
st.sidebar.header("⚙️ Settings")

# Distance scale
st.sidebar.subheader("📏 Distance Scale Settings")
enable_distance_scale = st.sidebar.checkbox("Enable real distance calculation", value=False)

if enable_distance_scale:
    st.sidebar.markdown("**Image Range (Latitude/Longitude)**")
    col_lat1, col_lat2 = st.sidebar.columns(2)
    with col_lat1:
        north_latitude = st.number_input("North Latitude", value=35.1, format="%.6f")
    with col_lat2:
        south_latitude = st.number_input("South Latitude", value=35.0, format="%.6f")
    col_lon1, col_lon2 = st.sidebar.columns(2)
    with col_lon1:
        west_longitude = st.number_input("West Longitude", value=135.0, format="%.6f")
    with col_lon2:
        east_longitude = st.number_input("East Longitude", value=135.1, format="%.6f")
    col_size1, col_size2 = st.sidebar.columns(2)
    with col_size1:
        image_width_px = st.number_input("Width (px)", value=480)
    with col_size2:
        image_height_px = st.number_input("Height (px)", value=360)

# Skeletonization & curvature settings
st.sidebar.subheader("🧠 Graph Detection Parameters")
resize_enabled = st.sidebar.checkbox("Resize to 480x360", value=True)
curvature_threshold = st.sidebar.slider("Curvature node threshold (°)", 5.0, 45.0, 15.0, 1.0)
curvature_window = st.sidebar.slider("Curvature analysis window size", 3, 10, 5, 1)
max_jump_distance = st.sidebar.slider("Max jump distance (pixels)", 1, 5, 2)
min_intersection_transitions = st.sidebar.slider("Intersection detection threshold", 2, 5, 3)
min_node_area = st.sidebar.slider("Minimum node area", 1, 10, 1)

# File upload
uploaded_file = st.file_uploader("Upload map image (road-line type)", type=["png", "jpg", "jpeg"])


# --- Utility functions ---

def calculate_distance_scale(north_lat, south_lat, west_lon, east_lon, width_px, height_px):
    EARTH_RADIUS = 6371000
    center_lat = (north_lat + south_lat) / 2
    center_lat_rad = math.radians(center_lat)
    lon_diff = abs(east_lon - west_lon)
    lon_diff_rad = math.radians(lon_diff)
    distance_x_m = EARTH_RADIUS * lon_diff_rad * math.cos(center_lat_rad)
    meters_per_pixel_x = distance_x_m / width_px
    lat_diff = abs(north_lat - south_lat)
    lat_diff_rad = math.radians(lat_diff)
    distance_y_m = EARTH_RADIUS * lat_diff_rad
    meters_per_pixel_y = distance_y_m / height_px
    return (meters_per_pixel_x + meters_per_pixel_y) / 2


def high_quality_skeletonization(img):
    """Clean skeletonization with adaptive threshold"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    binary_bool = (binary > 128).astype(bool)
    skeleton_bool = skeletonize(binary_bool)
    return skeleton_bool.astype(np.uint8)


# --- Core graph detection function (with curvature integration) ---
def detect_and_build_graph_with_curvature(binary_img,
                                          curvature_threshold=15.0,
                                          curvature_window=5,
                                          max_jump=2,
                                          min_transitions=3,
                                          min_node_area=1):
    H, W = binary_img.shape
    neighbors_coord = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
                       (1, 1), (1, 0), (1, -1), (0, -1)]

    # 1️⃣ トポロジ特徴点 (交差点・端点)
    feature_map = np.zeros_like(binary_img)
    feature_pixels = {}
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if binary_img[y, x] == 1:
                neighbors = [binary_img[y + dy, x + dx] for dy, dx in neighbors_coord]
                transitions = sum(neighbors[i] == 0 and neighbors[(i + 1) % 8] == 1 for i in range(8))
                if transitions >= min_transitions:
                    feature_map[y, x] = 1
                    feature_pixels[(y, x)] = 0  # intersection
                elif transitions == 1:
                    feature_map[y, x] = 1
                    feature_pixels[(y, x)] = 2  # endpoint

    # 2️⃣ 曲率ノード検出 (PCA法)
    ys, xs = np.nonzero(binary_img)
    curvature_map = np.zeros_like(binary_img, dtype=np.uint8)
    for (y, x) in zip(ys, xs):
        y1, y2 = max(0, y - curvature_window), min(H, y + curvature_window + 1)
        x1, x2 = max(0, x - curvature_window), min(W, x + curvature_window + 1)
        region = binary_img[y1:y2, x1:x2]
        pts = np.column_stack(np.nonzero(region))
        if len(pts) < 3:
            continue
        pca = PCA(n_components=2)
        pca.fit(pts - np.mean(pts, axis=0))
        dir1 = pca.components_[0]
        ny, nx = y + 1, x
        if ny < H and binary_img[ny, nx] == 1:
            y1b, y2b = max(0, ny - curvature_window), min(H, ny + curvature_window + 1)
            x1b, x2b = max(0, nx - curvature_window), min(W, nx + curvature_window + 1)
            region_b = binary_img[y1b:y2b, x1b:x2b]
            pts_b = np.column_stack(np.nonzero(region_b))
            if len(pts_b) >= 3:
                pca_b = PCA(n_components=2)
                pca_b.fit(pts_b - np.mean(pts_b, axis=0))
                dir2 = pca_b.components_[0]
                angle = np.degrees(np.arccos(np.clip(np.dot(dir1, dir2), -1, 1)))
                if angle > curvature_threshold:
                    curvature_map[y, x] = 1
                    feature_map[y, x] = 1
                    feature_pixels[(y, x)] = 1  # curve node

    # 3️⃣ ノードクラスタ統合
    labeled_img = label(feature_map, connectivity=2)
    regions = regionprops(labeled_img)
    nodes = {}
    coord_to_node_id = np.full((H, W), -1, dtype=int)
    node_id = 1
    for region in regions:
        if region.area < min_node_area:
            continue
        cy, cx = region.centroid
        types = [feature_pixels.get((y, x), -1) for y, x in region.coords]
        major_type = collections.Counter(types).most_common(1)[0][0]
        nodes[node_id] = {'pos': (int(cx), int(cy)),
                          'type': major_type,
                          'adj': [],
                          'coords': list(region.coords)}
        for (y, x) in region.coords:
            coord_to_node_id[y, x] = node_id
        node_id += 1

    # 4️⃣ エッジ構築
    edges = set()
    edge_visited = np.full((H, W), -1, dtype=int)
    edge_id = 0
    for nid, nd in nodes.items():
        for (y, x) in nd['coords']:
            for dy, dx in neighbors_coord:
                ny, nx = y + dy, x + dx
                if not (0 <= ny < H and 0 <= nx < W):
                    continue
                if binary_img[ny, nx] != 1:
                    continue
                nid2 = coord_to_node_id[ny, nx]
                if nid2 != -1 and nid2 != nid:
                    ek = tuple(sorted((nid, nid2)))
                    if ek not in edges:
                        edges.add(ek)
                        nodes[nid]['adj'].append((nid2, 1))
                        nodes[nid2]['adj'].append((nid, 1))
    # ノード描画
    marked_img = cv2.cvtColor(binary_img * 255, cv2.COLOR_GRAY2BGR)
    for nid, nd in nodes.items():
        x, y = nd['pos']
        color = (255, 0, 0) if nd['type'] == 0 else (0, 255, 0) if nd['type'] == 1 else (0, 255, 255)
        cv2.circle(marked_img, (x, y), 4, color, -1)
    return nodes, edges, marked_img


def create_csv_data(nodes, edges, img_h, meters_per_pixel=None):
    node_rows, edge_rows = [], []
    type_labels = {0: 'Intersection', 1: 'Curve', 2: 'Endpoint'}
    for nid, nd in nodes.items():
        x, y = nd['pos']
        x_s, y_s = int(round(x - 240)), int(round(img_h / 2 - y))
        node_rows.append([nid, x_s, y_s, nd['type'], type_labels.get(nd['type'], 'Unknown')])
    eid = 1
    for n1, n2 in edges:
        length = 1
        if meters_per_pixel:
            dist = length * meters_per_pixel
            edge_rows.append([eid, n1, n2, length, f"{dist:.2f}"])
            eid += 1
            edge_rows.append([eid, n2, n1, length, f"{dist:.2f}"])
        else:
            edge_rows.append([eid, n1, n2, length])
            eid += 1
            edge_rows.append([eid, n2, n1, length])
            eid += 1
    return node_rows, edge_rows


# --- Main process ---
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.success("✅ Image uploaded successfully")

    if enable_distance_scale:
        m_per_px = calculate_distance_scale(
            north_latitude, south_latitude,
            west_longitude, east_longitude,
            image_width_px, image_height_px
        )
        st.info(f"1 px ≈ {m_per_px:.2f} m")

    if st.button("🚀 Generate Graph Data", type="primary"):
        with st.spinner("Processing..."):
            if resize_enabled:
                img = cv2.resize(img, (480, 360))
            skeleton = high_quality_skeletonization(img)
            nodes, edges, marked = detect_and_build_graph_with_curvature(
                skeleton,
                curvature_threshold,
                curvature_window,
                max_jump_distance,
                min_intersection_transitions,
                min_node_area
            )
            st.success(f"Nodes: {len(nodes)}, Edges: {len(edges)}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
            with col2:
                st.image(skeleton * 255, caption="Skeleton", use_container_width=True)
            with col3:
                st.image(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB), caption="Detected Graph", use_container_width=True)

            node_csv, edge_csv = create_csv_data(nodes, edges, 360, m_per_px if enable_distance_scale else None)
            n_df = pd.DataFrame(node_csv, columns=['id', 'x', 'y', 'type', 'label'])
            e_df = pd.DataFrame(edge_csv, columns=['eid', 'from', 'to', 'len', 'm'] if enable_distance_scale else ['eid', 'from', 'to', 'len'])
            st.download_button("⬇️ Download Node CSV", n_df.to_csv(index=False), "nodes.csv", "text/csv")
            st.download_button("⬇️ Download Edge CSV", e_df.to_csv(index=False), "edges.csv", "text/csv")

else:
    st.info("👆 Please upload a road-line image.")
