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

# ページ設定
st.set_page_config(
    page_title="画像グラフ生成アプリ",
    page_icon="📊",
    layout="wide"
)

# タイトル
st.title("📊 画像からグラフデータを生成")
st.markdown("画像をアップロードして、スケルトン化とグラフ構築を行い、CSVデータを生成します。")

# サイドバー設定
st.sidebar.header("⚙️ 設定")
st.sidebar.subheader("📏 距離スケール設定")
enable_distance_scale = st.sidebar.checkbox("実距離計算を有効化", value=False)

if enable_distance_scale:
    st.sidebar.markdown("**画像の範囲（緯度経度）**")
    col_lat1, col_lat2 = st.sidebar.columns(2)
    with col_lat1:
        north_latitude = st.number_input("北緯度", value=35.1, format="%.6f", step=0.000001)
    with col_lat2:
        south_latitude = st.number_input("南緯度", value=35.0, format="%.6f", step=0.000001)
    col_lon1, col_lon2 = st.sidebar.columns(2)
    with col_lon1:
        west_longitude = st.number_input("西経度", value=135.0, format="%.6f", step=0.000001)
    with col_lon2:
        east_longitude = st.number_input("東経度", value=135.1, format="%.6f", step=0.000001)
    st.sidebar.markdown("**画像サイズ（ピクセル）**")
    col_size1, col_size2 = st.sidebar.columns(2)
    with col_size1:
        image_width_px = st.number_input("幅", value=480, min_value=1)
    with col_size2:
        image_height_px = st.number_input("高さ", value=360, min_value=1)

st.sidebar.subheader("画像処理")
resize_enabled = st.sidebar.checkbox("画像を480x360にリサイズ", value=True)
st.sidebar.subheader("グラフ構築")
curvature_threshold = st.sidebar.slider("曲率分割閾値", 1.0, 20.0, 10.0, 0.5)
max_jump_distance = st.sidebar.slider("最大ジャンプ距離", 1, 5, 2)
min_intersection_transitions = st.sidebar.slider("交差点検出閾値", 2, 5, 3)
min_node_area = st.sidebar.slider("最小ノード面積", 1, 10, 1)

uploaded_file = st.file_uploader("画像ファイルをアップロード", type=['png', 'jpg', 'jpeg'])


# --------------------------
# 共通関数群
# --------------------------

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
    return meters_per_pixel_x, meters_per_pixel_y, (meters_per_pixel_x + meters_per_pixel_y) / 2


def resize_image(img, target_width=480, target_height=360):
    resized_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized_img, img.shape[0], img.shape[1]


def refine_skeleton_branches(skeleton):
    H, W = skeleton.shape
    refined = skeleton.copy()
    neighbors_8 = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
                   (1, 1), (1, 0), (1, -1), (0, -1)]
    endpoints = []
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if skeleton[y, x] == 1:
                neighbors = [skeleton[y + dy, x + dx] for dy, dx in neighbors_8]
                if sum(neighbors) == 1:
                    endpoints.append((y, x))
    for sy, sx in endpoints:
        branch = [(sy, sx)]
        y, x = sy, sx
        for _ in range(5):
            nexts = [(y + dy, x + dx) for dy, dx in neighbors_8
                     if 0 <= y + dy < H and 0 <= x + dx < W and skeleton[y + dy, x + dx] == 1]
            if len(nexts) != 1:
                break
            y, x = nexts[0]
            branch.append((y, x))
        if len(branch) < 5:
            for by, bx in branch:
                refined[by, bx] = 0
    return refined


def high_quality_skeletonization(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    dilated = cv2.dilate(cleaned, np.ones((3, 3), np.uint8), iterations=1)
    skeleton = skeletonize((dilated > 128).astype(bool)).astype(np.uint8)
    filtered = np.zeros_like(skeleton)
    for region in regionprops(label(skeleton)):
        if region.area >= 5:
            for y, x in region.coords:
                filtered[y, x] = 1
    return refine_skeleton_branches(filtered), (filtered * 255).astype(np.uint8)


# --------------------------
# ノード統合関数（今回追加）
# --------------------------

def merge_nodes_by_position(nodes):
    type_rank = {0: 3, 1: 2, 2: 1, 3: 0}
    pos_to_newid, old_to_new, new_nodes = {}, {}, {}

    # ノード統合（同一座標）
    for old_id in sorted(nodes.keys()):
        d = nodes[old_id]
        pos = tuple(d['pos'])
        if pos not in pos_to_newid:
            new_id = len(pos_to_newid) + 1
            pos_to_newid[pos] = new_id
            new_nodes[new_id] = {
                'pos': pos, 'type': d['type'],
                'adj': [], 'coords': list(d.get('coords', []))
            }
        else:
            nid = pos_to_newid[pos]
            if type_rank[d['type']] > type_rank[new_nodes[nid]['type']]:
                new_nodes[nid]['type'] = d['type']
            new_nodes[nid]['coords'].extend(d.get('coords', []))
        old_to_new[old_id] = pos_to_newid[pos]

    # エッジ再構築
    length_map = {}
    for oid, d in nodes.items():
        a = old_to_new[oid]
        for nb, length in d['adj']:
            b = old_to_new[nb]
            if a == b:
                continue
            if a > b:
                a, b = b, a
            key = (a, b)
            if key not in length_map or length < length_map[key]:
                length_map[key] = length

    for (a, b), l in length_map.items():
        new_nodes[a]['adj'].append((b, l))
        new_nodes[b]['adj'].append((a, l))
    return new_nodes, set(length_map.keys()), old_to_new


# --------------------------
# グラフ検出関数（省略済みロジック）
# --------------------------
def detect_and_build_graph(binary_img, curvature_threshold, max_jump, min_transitions, min_area):
    H, W = binary_img.shape
    feature_map = np.zeros_like(binary_img)
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
                 (1, 1), (1, 0), (1, -1), (0, -1)]
    feature_pixels = {}
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if binary_img[y, x] == 1:
                n = [binary_img[y + dy, x + dx] for dy, dx in neighbors]
                t = sum(n[i] == 0 and n[(i + 1) % 8] == 1 for i in range(8))
                if t >= min_transitions:
                    feature_map[y, x] = 1
                    feature_pixels[(y, x)] = 0
                elif t == 1:
                    feature_map[y, x] = 1
                    feature_pixels[(y, x)] = 2
    labeled = label(feature_map)
    nodes = {}
    coord_to_node_id = np.full((H, W), -1, int)
    nid = 1
    for r in regionprops(labeled):
        if r.area < min_area:
            continue
        cy, cx = map(int, r.centroid)
        nodes[nid] = {'pos': (cx, cy), 'type': 0, 'adj': [], 'coords': r.coords}
        for y, x in r.coords:
            coord_to_node_id[y, x] = nid
        nid += 1
    if not nodes:
        return None, None, None
    return nodes, set(), (binary_img * 255).astype(np.uint8)


# --------------------------
# CSV生成
# --------------------------
def create_csv_data(nodes, edges, h, mpp=None):
    node_data, edge_data = [], []
    for i, d in nodes.items():
        x, y = d['pos']
        xs, ys = x - 240, h / 2 - y
        node_data.append([i, int(xs), int(ys), d['type']])
    eid = 1
    for a, b in edges:
        l = min([l for n, l in nodes[a]['adj'] if n == b] or [0])
        if mpp:
            edge_data.append([eid, a, b, l, f"{l * mpp:.2f}"])
        else:
            edge_data.append([eid, a, b, l])
        eid += 1
    return node_data, edge_data


def create_csv_file(data, header):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(data)
    return buf.getvalue()


# --------------------------
# メイン処理
# --------------------------
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.success("✅ 画像をアップロードしました")

    if enable_distance_scale:
        mx, my, mavg = calculate_distance_scale(
            north_latitude, south_latitude,
            west_longitude, east_longitude,
            image_width_px, image_height_px
        )
        st.info(f"📏距離スケール: 横={mx:.2f}m/px, 縦={my:.2f}m/px, 平均={mavg:.2f}m/px")

    if st.button("🚀 グラフデータを生成"):
        with st.spinner("処理中..."):
            img, _, _ = resize_image(img, 480, 360)
            skel, vis = high_quality_skeletonization(img)
            nodes, edges, marked = detect_and_build_graph(
                skel, curvature_threshold, max_jump_distance,
                min_intersection_transitions, min_node_area
            )

            if nodes is None:
                st.error("検出失敗。パラメータを調整してください。")
            else:
                # === ここで統合 ===
                nodes, edges, mapping = merge_nodes_by_position(nodes)

                st.success(f"ノード数: {len(nodes)}, エッジ数: {len(edges)}")
                st.image(vis, caption="スケルトン", use_container_width=True)

                if enable_distance_scale:
                    n, e = create_csv_data(nodes, edges, 360, mavg)
                else:
                    n, e = create_csv_data(nodes, edges, 360)

                st.download_button("ノードCSV", create_csv_file(n, ['id', 'x', 'y', 'type']),
                                   file_name="nodes.csv", mime="text/csv")
                if enable_distance_scale:
                    st.download_button("エッジCSV", create_csv_file(e, ['id', 'from', 'to', 'len', 'm']),
                                       file_name="edges.csv", mime="text/csv")
                else:
                    st.download_button("エッジCSV", create_csv_file(e, ['id', 'from', 'to', 'len']),
                                       file_name="edges.csv", mime="text/csv")
else:
    st.info("👆 画像をアップロードしてください。")

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
