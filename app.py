import streamlit as st
import cv2
import numpy as np
import math
import csv
import io
from PIL import Image

st.set_page_config(
    page_title="Road Network Extraction Tool",
    page_icon="🗺️",
    layout="wide"
)

# Title
st.title("🗺️ Road Network Extraction Tool")
st.markdown("Automatically extract road networks from map images and generate CSV data for Scratch")

# Sidebar - Settings
st.sidebar.header("⚙️ Parameter Settings")

corner_threshold = st.sidebar.slider(
    "Corner Detection Angle (degrees)",
    min_value=30,
    max_value=120,
    value=60,
    step=5,
    help="Detect corners with angles greater than this value"
)

cluster_radius = st.sidebar.slider(
    "Node Merge Radius (pixels)",
    min_value=5,
    max_value=30,
    value=15,
    step=1,
    help="Merge nearby nodes within this distance"
)

max_distance = st.sidebar.slider(
    "Edge Maximum Distance (pixels)",
    min_value=50,
    max_value=400,
    value=200,
    step=10,
    help="Do not connect nodes farther than this distance"
)

angle_tolerance = st.sidebar.slider(
    "Angle Tolerance (degrees)",
    min_value=10,
    max_value=45,
    value=25,
    step=5,
    help="Minimum angle difference between roads from the same node"
)

bright_threshold = st.sidebar.slider(
    "White Area Detection Threshold",
    min_value=100,
    max_value=250,
    value=180,
    step=10,
    help="Pixels brighter than this value are treated as obstacles"
)

# Scratch coordinate settings
st.sidebar.markdown("---")
st.sidebar.subheader("🎮 Scratch Coordinate Settings")
scratch_width = st.sidebar.number_input("Scratch Width", value=480, step=10)
scratch_height = st.sidebar.number_input("Scratch Height", value=360, step=10)

SCRATCH_XRANGE = (-scratch_width//2, scratch_width//2)
SCRATCH_YRANGE = (-scratch_height//2, scratch_height//2)

# File upload
uploaded_file = st.file_uploader(
    "Upload Map Image (PNG, JPG)",
    type=["png", "jpg", "jpeg"]
)

def skeletonize(img_bin):
    """Skeletonize the image"""
    skel = np.zeros_like(img_bin)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    while True:
        open_img = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img_bin, open_img)
        eroded = cv2.erode(img_bin, element)
        skel = cv2.bitwise_or(skel, temp)
        img_bin = eroded.copy()
        if cv2.countNonZero(img_bin) == 0:
            break
    return skel

def get_neighbor_directions(skel, y, x):
    """Get direction vectors of adjacent white pixels"""
    directions = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1]:
                if skel[ny, nx] == 255:
                    directions.append((dy, dx))
    return directions

def calculate_angle_between_vectors(v1, v2):
    """Calculate angle between two vectors (degrees)"""
    angle1 = math.degrees(math.atan2(v1[0], v1[1]))
    angle2 = math.degrees(math.atan2(v2[0], v2[1]))
    diff = abs(angle1 - angle2)
    if diff > 180:
        diff = 360 - diff
    return diff

def is_connected(img, y1, x1, y2, x2, max_dist, bright_thresh):
    """Check if two points can be connected"""
    h, w = img.shape
    dy, dx = y2 - y1, x2 - x1
    dist = int(math.hypot(dx, dy))
    if dist < 5 or dist > max_dist:
        return False
    step_y, step_x = dy / dist, dx / dist
    for i in range(dist):
        py = int(round(y1 + step_y * i))
        px = int(round(x1 + step_x * i))
        if not (0 <= py < h and 0 <= px < w):
            return False
        if np.mean(img[max(0, py-1):py+2, max(0, px-1):px+2]) > bright_thresh:
            return False
    return True

def to_scratch(x, y, w, h):
    """Convert image coordinates to Scratch coordinates"""
    x_s = (x / w) * (SCRATCH_XRANGE[1] - SCRATCH_XRANGE[0]) + SCRATCH_XRANGE[0]
    y_s = SCRATCH_YRANGE[1] - (y / h) * (SCRATCH_YRANGE[1] - SCRATCH_YRANGE[0])
    return round(x_s, 2), round(y_s, 2)

def process_image(img_array, params):
    """Process image to extract nodes and edges"""
    # Convert to grayscale
    if len(img_array.shape) == 3:
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img = img_array
    
    h0, w0 = img.shape
    
    # Binarization and skeletonization
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    skeleton = skeletonize(binary)
    h, w = skeleton.shape
    
    # Node detection
    nodes = []
    node_types = []
    
    for y in range(1, h-1):
        for x in range(1, w-1):
            if skeleton[y, x] != 255:
                continue
            
            directions = get_neighbor_directions(skeleton, y, x)
            num_neighbors = len(directions)
            
            if num_neighbors == 1:
                nodes.append((y, x))
                node_types.append('endpoint')
            elif num_neighbors >= 3:
                nodes.append((y, x))
                node_types.append('junction')
            elif num_neighbors == 2:
                angle_diff = calculate_angle_between_vectors(directions[0], directions[1])
                if angle_diff > params['corner_threshold']:
                    nodes.append((y, x))
                    node_types.append('corner')
    
    # Node merging
    merged = []
    merged_types = []
    used = np.zeros(len(nodes), bool)
    
    for i, (y1, x1) in enumerate(nodes):
        if used[i]:
            continue
        cluster = [(y1, x1)]
        cluster_types = [node_types[i]]
        used[i] = True
        
        for j, (y2, x2) in enumerate(nodes):
            if not used[j] and i != j:
                if (y1 - y2)**2 + (x1 - x2)**2 < params['cluster_radius']**2:
                    cluster.append((y2, x2))
                    cluster_types.append(node_types[j])
                    used[j] = True
        
        y_mean = np.mean([p[0] for p in cluster])
        x_mean = np.mean([p[1] for p in cluster])
        merged.append((y_mean, x_mean))
        
        if 'junction' in cluster_types:
            merged_types.append('junction')
        elif 'corner' in cluster_types:
            merged_types.append('corner')
        else:
            merged_types.append('endpoint')
    
    centroids = merged
    
    # Edge detection
    edges = []
    for i in range(len(centroids)):
        y1, x1 = centroids[i]
        connected_dirs = []
        for j in range(i+1, len(centroids)):
            y2, x2 = centroids[j]
            if not is_connected(img, y1, x1, y2, x2, params['max_distance'], params['bright_threshold']):
                continue
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if any(abs((angle - a + 180) % 360 - 180) < params['angle_tolerance'] for a in connected_dirs):
                continue
            connected_dirs.append(angle)
            edges.append((i, j))
    
    # Visualization
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    line_layer = overlay.copy()
    
    for e in edges:
        y1, x1 = centroids[e[0]]
        y2, x2 = centroids[e[1]]
        cv2.line(line_layer, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
    
    overlay = cv2.addWeighted(line_layer, 0.7, overlay, 0.3, 0)
    
    for i, (y, x) in enumerate(centroids, start=1):
        node_type = merged_types[i-1]
        if node_type == 'endpoint':
            color = (255, 0, 0)
            radius = 5
        elif node_type == 'junction':
            color = (0, 255, 0)
            radius = 6
        else:
            color = (0, 165, 255)
            radius = 4
        
        cv2.circle(overlay, (int(x), int(y)), radius, color, -1)
        cv2.putText(overlay, str(i), (int(x)+5, int(y)-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    
    return {
        'skeleton': skeleton,
        'overlay': overlay,
        'centroids': centroids,
        'merged_types': merged_types,
        'edges': edges,
        'dimensions': (h, w)
    }

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Start processing button
    if st.button("🚀 Start Processing", type="primary"):
        with st.spinner("Processing..."):
            params = {
                'corner_threshold': corner_threshold,
                'cluster_radius': cluster_radius,
                'max_distance': max_distance,
                'angle_tolerance': angle_tolerance,
                'bright_threshold': bright_threshold
            }
            
            result = process_image(img_array, params)
            
            # Display results
            st.success("✅ Processing Complete!")
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Nodes", len(result['centroids']))
            with col2:
                st.metric("Endpoints", result['merged_types'].count('endpoint'))
            with col3:
                st.metric("Junctions", result['merged_types'].count('junction'))
            with col4:
                st.metric("Corners", result['merged_types'].count('corner'))
            
            st.metric("Total Edges", len(result['edges']))
            
            # Image display
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Skeleton Image")
                st.image(result['skeleton'], use_container_width=True)
            with col2:
                st.subheader("Detection Result")
                st.image(cv2.cvtColor(result['overlay'], cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # CSV generation
            h, w = result['dimensions']
            
            # nodes.csv
            nodes_csv = io.StringIO()
            writer = csv.writer(nodes_csv)
            writer.writerow(["id", "x", "y", "type"])
            for i, (y, x) in enumerate(result['centroids'], start=1):
                xs, ys = to_scratch(x, y, w, h)
                writer.writerow([i, xs, ys, result['merged_types'][i-1]])
            
            # edges.csv
            edges_csv = io.StringIO()
            writer = csv.writer(edges_csv)
            writer.writerow(["source_id", "target_id"])
            for e in result['edges']:
                writer.writerow([e[0]+1, e[1]+1])
                writer.writerow([e[1]+1, e[0]+1])
            
            # Download buttons
            st.subheader("📥 Download")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    label="📄 nodes.csv",
                    data=nodes_csv.getvalue(),
                    file_name="nodes.csv",
                    mime="text/csv"
                )
            
            with col2:
                st.download_button(
                    label="📄 edges.csv",
                    data=edges_csv.getvalue(),
                    file_name="edges.csv",
                    mime="text/csv"
                )
            
            with col3:
                # Download result image as PNG
                is_success, buffer = cv2.imencode(".png", result['overlay'])
                if is_success:
                    st.download_button(
                        label="🖼️ Result Image",
                        data=buffer.tobytes(),
                        file_name="road_network.png",
                        mime="image/png"
                    )

else:
    st.info("👆 Please upload a map image")
    
    # Instructions
    with st.expander("📖 How to Use"):
        st.markdown("""
        ### Usage Steps
        1. **Upload Map Image**: PNG/JPG format road map image
        2. **Adjust Parameters**: Configure detection settings in the left sidebar
        3. **Start Processing**: Click the button to begin analysis
        4. **Download Results**: Save CSV files and result image
        
        ### Node Types
        - 🔵 **Endpoint**: Road terminus
        - 🟢 **Junction**: Intersection of 3 or more roads
        - 🟠 **Corner**: Road turning point
        
        ### Output Files
        - **nodes.csv**: Node coordinates and types
        - **edges.csv**: Connection information between nodes
        - **Result Image**: Visualization of detection results
        """)

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using Streamlit")
