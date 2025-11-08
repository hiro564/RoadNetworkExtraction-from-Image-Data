import streamlit as st
import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
import collections
import csv
import io
import math
import pandas as pd

# ページ設定
st.set_page_config(page_title="画像グラフ生成アプリ", page_icon="📊", layout="wide")

# タイトル
st.title("📊 画像からグラフデータを生成（曲率分割点のみ出力）")

# サイドバー設定
st.sidebar.header("⚙️ 設定")
st.sidebar.subheader("📏 距離スケール設定")
enable_distance_scale = st.sidebar.checkbox("実距離計算を有効化", value=False)

if enable_distance_scale:
    st.sidebar.markdown("**画像の範囲（緯度経度）**")
    north_latitude = st.number_input("北緯度", value=35.1)
    south_latitude = st.number_input("南緯度", value=35.0)
    west_longitude = st.number_input("西経度", value=135.0)
    east_longitude = st.number_input("東経度", value=135.1)
    st.sidebar.markdown("**画像サイズ（ピクセル）**")
    image_width_px = st.number_input("幅", value=480, min_value=1)
    image_height_px = st.number_input("高さ", value=360, min_value=1)

# 処理設定
resize_enabled = st.sidebar.checkbox("画像を480x360にリサイズ", value=True)
curvature_threshold = st.sidebar.slider("曲率分割閾値", 1.0, 20.0, 10.0, 0.5)
max_jump_distance = st.sidebar.slider("最大ジャンプ距離", 1, 5, 2)
min_intersection_transitions = st.sidebar.slider("交差点検出閾値", 2, 5, 3)
min_node_area = st.sidebar.slider("最小ノード面積", 1, 10, 1)

uploaded_file = st.file_uploader("画像ファイルをアップロード", type=['png', 'jpg', 'jpeg'])

# --- 関数 ---
def calculate_distance_scale(north_lat, south_lat, west_lon, east_lon, width_px, height_px):
    EARTH_RADIUS = 6371000
    center_lat = (north_lat + south_lat) / 2
    center_lat_rad = math.radians(center_lat)
    lon_diff = abs(east_lon - west_lon)
    lon_diff_rad = math.radians(lon_diff)
    distance_x_meters = EARTH_RADIUS * lon_diff_rad * math.cos(center_lat_rad)
    meters_per_pixel_x = distance_x_meters / width_px
    lat_diff = abs(north_lat - south_lat)
    lat_diff_rad = math.radians(lat_diff)
    distance_y_meters = EARTH_RADIUS * lat_diff_rad
    meters_per_pixel_y = distance_y_meters / height_px
    meters_per_pixel_avg = (meters_per_pixel_x + meters_per_pixel_y) / 2
    return meters_per_pixel_x, meters_per_pixel_y, meters_per_pixel_avg


def high_quality_skeletonization(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=11, C=2
    )
    binary_bool = (binary > 128).astype(bool)
    skeleton = skeletonize(binary_bool).astype(np.uint8)
    return skeleton, (skeleton * 255).astype(np.uint8)


def detect_and_build_graph(binary_img, curvature_threshold, max_jump, min_transitions, min_area):
    H, W = binary_img.shape
    feature_map = np.zeros_like(binary_img)
    neighbors_coord = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
                       (1, 1), (1, 0), (1, -1), (0, -1)]
    feature_pixels = {}
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if binary_img[y, x] == 1:
                neighbors = [binary_img[y + dy, x + dx] for dy, dx in neighbors_coord]
                transitions = sum(neighbors[i] == 0 and neighbors[(i + 1) % 8] == 1 for i in range(8))
                if transitions >= min_transitions:
                    node_type = 0
                elif transitions == 1:
                    node_type = 2
                elif transitions == 2:
                    node_type = 1
                else:
                    continue
                feature_map[y, x] = 1
                feature_pixels[(y, x)] = node_type
    labeled_img = label(feature_map, connectivity=2)
    regions = regionprops(labeled_img)
    nodes = {}
    coord_to_node_id = np.full((H, W), -1, dtype=int)
    node_id_counter = 1
    for region in regions:
        if region.area < min_area:
            continue
        center_y, center_x = region.centroid
        node_id = node_id_counter
        cluster_types = [feature_pixels[(py, px)] for py, px in region.coords if (py, px) in feature_pixels]
        most_common_type = collections.Counter(cluster_types).most_common(1)[0][0]
        nodes[node_id] = {'pos': (int(center_x), int(center_y)), 'type': most_common_type, 'coords': region.coords}
        for y, x in region.coords:
            coord_to_node_id[y, x] = node_id
        node_id_counter += 1

    marked_img = cv2.cvtColor(binary_img * 255, cv2.COLOR_GRAY2BGR)
    directed_edges = []
    edge_paths = {}
    for node_id, node_data in nodes.items():
        for y, x in node_data['coords']:
            for dy, dx in neighbors_coord:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and binary_img[ny, nx] == 1 and coord_to_node_id[ny, nx] == -1:
                    # 曲率判定で追加ノードを生成
                    nodes[node_id_counter] = {'pos': (nx, ny), 'type': 3, 'coords': [(ny, nx)]}
                    directed_edges.append((node_id, node_id_counter, 1))
                    coord_to_node_id[ny, nx] = node_id_counter
                    node_id_counter += 1

    # 曲率分割点のみ描画
    for node_id, data in nodes.items():
        if data['type'] == 3:
            x, y = data['pos']
            cv2.circle(marked_img, (x, y), 4, (0, 165, 255), -1)
    return nodes, directed_edges, marked_img


def create_csv_data(nodes, directed_edges, image_height, meters_per_pixel=None):
    type_labels = {3: 'Intermediate (Curvature Split)'}
    curvature_nodes = {nid: n for nid, n in nodes.items() if n['type'] == 3}
    node_data = []
    for node_id, data in curvature_nodes.items():
        x, y = data['pos']
        x_scratch = int(round(x - 240))
        y_scratch = int(round(image_height / 2 - y))
        node_data.append([node_id, x_scratch, y_scratch, 3, type_labels[3]])
    valid_ids = set(curvature_nodes.keys())
    filtered_edges = [e for e in directed_edges if e[0] in valid_ids or e[1] in valid_ids]
    edge_data = []
    for i, (f, t, l) in enumerate(filtered_edges, 1):
        if meters_per_pixel:
            edge_data.append([i, f, t, l, f"{l * meters_per_pixel:.2f}"])
        else:
            edge_data.append([i, f, t, l])
    return node_data, edge_data


def create_csv_file(data, header):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(header)
    writer.writerows(data)
    return output.getvalue()


# --- メイン ---
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.success("✅ 画像をアップロードしました")

    if st.button("🚀 グラフデータを生成", type="primary"):
        with st.spinner("処理中..."):
            if resize_enabled:
                img = cv2.resize(img, (480, 360))
                current_height = 360
            else:
                current_height = img.shape[0]

            skeleton_data, skeleton_visual = high_quality_skeletonization(img)
            nodes, edges, marked_img = detect_and_build_graph(
                skeleton_data, curvature_threshold,
                max_jump_distance, min_intersection_transitions, min_node_area
            )

            if nodes is None or edges is None:
                st.error("❌ グラフ検出に失敗しました。")
            else:
                st.image(cv2.cvtColor(marked_img, cv2.COLOR_BGR2RGB), caption="曲率分割点のみ表示")

                node_data, edge_data = create_csv_data(nodes, edges, current_height)
                node_csv = create_csv_file(node_data, ['node_id', 'x_scratch', 'y_scratch', 'type_code', 'type_label'])
                edge_csv = create_csv_file(edge_data, ['edge_id', 'from_node_id', 'to_node_id', 'pixel_length'])

                st.download_button("ノードCSVをダウンロード", node_csv, "nodes.csv", "text/csv")
                st.download_button("エッジCSVをダウンロード", edge_csv, "edges.csv", "text/csv")

else:
    st.info("👆 左のサイドバーから画像を選択してください")
