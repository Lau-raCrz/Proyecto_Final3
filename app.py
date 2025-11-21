import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time
from collections import deque

# ============================================
# CONFIGURACIÓN
# ============================================
st.set_page_config(
    page_title="Detector YOLO - Herramientas y Personas",
    page_icon="🔍",
    layout="wide"
)

MODEL_PATH = "runs/detect/train4/weights/best.pt"
POSE_MODEL_PATH = "yolov8n-pose.pt"

CLASS_NAMES = [
    "Capacitor_electronico",
    "Cautin_para_soldar",
    "Destornillador_electronica",
    "Fuente_de_alimentacion_dc",
    "Multimetro_electronico",
    "Osciloscopio",
    "Protoboard",
    "Raspberry",
    "Resistencia_electronica",
    "Transformador_electronico"
]

# ============================================
# TRACKER DE PERSONAS CON VELOCIDAD
# ============================================
class PersonTracker:
    def __init__(self):
        self.persons = {}
        self.next_id = 1
        self.max_distance = 150
        self.alpha = 0.2

    def update(self, center, frame_number, time_diff):
        min_dist = float('inf')
        matched_id = None
        
        for pid, data in self.persons.items():
            if frame_number - data['last_seen'] < 30:
                prev_x, prev_y = data['position']
                dist = np.sqrt((center[0]-prev_x)**2 + (center[1]-prev_y)**2)
                if dist < min_dist and dist < self.max_distance:
                    min_dist = dist
                    matched_id = pid
        
        if matched_id:
            prev_vel_raw = self.persons[matched_id]['velocity_raw']
            vel_raw = min_dist / time_diff if time_diff > 0 else 0
            vel_suave = (self.alpha * vel_raw) + ((1 - self.alpha) * prev_vel_raw)
            vel_norm = min(10, vel_suave / 25)

            self.persons[matched_id] = {
                'position': center,
                'velocity_raw': vel_suave,
                'velocity_norm': vel_norm,
                'last_seen': frame_number
            }
            return matched_id, vel_norm
        else:
            new_id = self.next_id
            self.next_id += 1
            self.persons[new_id] = {
                'position': center,
                'velocity_raw': 0,
                'velocity_norm': 0,
                'last_seen': frame_number
            }
            return new_id, 0

    def cleanup(self, frame_number, max_frames=60):
        eliminar = [pid for pid, data in self.persons.items()
                    if frame_number - data['last_seen'] > max_frames]
        for pid in eliminar:
            del self.persons[pid]

@st.cache_resource
def load_models():
    model = YOLO(MODEL_PATH)
    pose_model = YOLO(POSE_MODEL_PATH)
    return model, pose_model

@st.cache_resource
def get_tracker():
    return PersonTracker()

def get_velocity_color(vel_norm):
    if vel_norm < 3:
        return (50, 255, 50), "LENTO"
    elif vel_norm < 6:
        return (50, 255, 255), "NORMAL"
    else:
        return (50, 50, 255), "RÁPIDO"

def draw_person_info(frame, cx, cy, pid, vel):
    color, status = get_velocity_color(vel)
    
    cv2.circle(frame, (cx, cy), 18, (255, 255, 255), -1)
    cv2.circle(frame, (cx, cy), 15, color, -1)
    cv2.circle(frame, (cx, cy), 18, color, 3)
    
    panel_x = cx - 100
    panel_y = cy - 115
    w, h = 200, 95
    
    if panel_x < 0:
        panel_x = cx + 20
    if panel_y < 0:
        panel_y = cy + 20
    
    cv2.rectangle(frame, (panel_x - 3, panel_y - 3),
                 (panel_x + w + 3, panel_y + h + 3), (0, 0, 0), -1)
    cv2.rectangle(frame, (panel_x, panel_y),
                 (panel_x + w, panel_y + h), (40, 40, 40), -1)
    cv2.rectangle(frame, (panel_x, panel_y),
                 (panel_x + w, panel_y + h), color, 4)
    
    cv2.rectangle(frame, (panel_x, panel_y),
                 (panel_x + w, panel_y + 30), (60, 60, 60), -1)
    
    cv2.putText(frame, f"PERSONA #{pid}", (panel_x + 15, panel_y + 22),
               cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.putText(frame, f"Vel: {vel:.1f}/10",
               (panel_x + 15, panel_y + 55),
               cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
    
    cv2.putText(frame, status, (panel_x + 15, panel_y + 82),
               cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 2)

def draw_pose_skeleton(frame, keypoints):
    skeleton = [
        [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
        [5, 11], [6, 12], [11, 12], [11, 13], 
        [13, 15], [12, 14], [14, 16]
    ]
    
    for (idx1, idx2) in skeleton:
        if idx1 < len(keypoints) and idx2 < len(keypoints):
            x1, y1 = keypoints[idx1]
            x2, y2 = keypoints[idx2]
            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                         (255, 100, 50), 3)
    
    for (x, y) in keypoints:
        if x > 0 and y > 0:
            cv2.circle(frame, (int(x), int(y)), 4, (50, 255, 255), -1)

def calculate_center_pose(keypoints, width, height):
    valid_points = []
    for idx in [5, 6, 11, 12]:
        if idx < len(keypoints):
            x, y = keypoints[idx]
            if x > 0 and y > 0:
                valid_points.append((x, y))
    
    if len(valid_points) == 0:
        return int(width / 2), int(height / 2)
    
    cx = int(sum(p[0] for p in valid_points) / len(valid_points))
    cy = int(sum(p[1] for p in valid_points) / len(valid_points))
    return cx, cy

def process_yolo_detection(frame, model):
    results = model(frame, conf=0.5, verbose=False)
    
    height, width = frame.shape[:2]
    frame_result = frame.copy()
    detections = []
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            w = int((x2 - x1) * 0.75)
            h = int((y2 - y1) * 0.75)
            
            x1 = max(0, cx - w // 2)
            y1 = max(0, cy - h // 2)
            x2 = min(width - 1, cx + w // 2)
            y2 = min(height - 1, cy + h // 2)
            
            cv2.rectangle(frame_result, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            label = f"{CLASS_NAMES[cls]} ({conf*100:.0f}%)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            cv2.rectangle(frame_result, (x1, y1 - th - 10), 
                         (x1 + tw + 5, y1), (0, 0, 0), -1)
            cv2.putText(frame_result, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            detections.append({
                'class': CLASS_NAMES[cls],
                'confidence': conf,
                'bbox': (x1, y1, x2, y2)
            })
    
    return frame_result, detections

def process_pose_detection_static(frame, pose_model):
    results = pose_model(frame, conf=0.5, verbose=False)
    
    frame_result = frame.copy()
    persons = []
    
    for result in results:
        if result.keypoints is not None:
            kps = result.keypoints.xy
            for i in range(kps.shape[0]):
                keypoints = kps[i].cpu().numpy()
                draw_pose_skeleton(frame_result, keypoints)
                
                height, width = frame.shape[:2]
                cx, cy = calculate_center_pose(keypoints, width, height)
                
                cv2.circle(frame_result, (cx, cy), 12, (255, 255, 255), -1)
                cv2.circle(frame_result, (cx, cy), 10, (50, 255, 50), -1)
                
                persons.append({
                    'center': (cx, cy),
                    'keypoints': keypoints
                })
    
    return frame_result, persons

def process_pose_detection_live(frame, pose_model, tracker, frame_number, time_diff):
    results = pose_model(frame, conf=0.5, verbose=False)
    
    frame_result = frame.copy()
    persons = []
    
    for result in results:
        if result.keypoints is not None:
            kps = result.keypoints.xy
            for i in range(kps.shape[0]):
                keypoints = kps[i].cpu().numpy()
                draw_pose_skeleton(frame_result, keypoints)
                
                height, width = frame.shape[:2]
                cx, cy = calculate_center_pose(keypoints, width, height)
                
                person_id, velocity = tracker.update((cx, cy), frame_number, time_diff)
                draw_person_info(frame_result, cx, cy, person_id, velocity)
                
                persons.append({
                    'id': person_id,
                    'center': (cx, cy),
                    'velocity': velocity,
                    'keypoints': keypoints
                })
    
    tracker.cleanup(frame_number)
    
    return frame_result, persons

def main():
    st.title("🔍 Detector YOLO - Herramientas y Personas")
    st.markdown("---")
    
    with st.spinner("Cargando modelos..."):
        model, pose_model = load_models()
        tracker = get_tracker()
    
    mode = st.sidebar.selectbox(
        "Modo de detección",
        ["📷 Subir Imagen", "🎥 Cámara en Vivo"]
    )
    
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("Opciones")
    detect_tools = st.sidebar.checkbox("Detectar Herramientas", value=True)
    detect_persons = st.sidebar.checkbox("Detectar Personas", value=True)
    
    if mode == "📷 Subir Imagen":
        st.header("📷 Subir Imagen")
        st.info("ℹ️ La velocidad solo se calcula en modo cámara en vivo")
        
        uploaded_file = st.file_uploader(
            "Sube una imagen (JPG, PNG, JPEG)",
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image_np
            
            st.subheader("Imagen Original")
            st.image(image, use_container_width=True)
            
            with st.spinner("Procesando..."):
                col1, col2 = st.columns(2)
                
                if detect_tools:
                    with col1:
                        st.subheader("🔧 Herramientas Detectadas")
                        frame_tools, detections = process_yolo_detection(
                            image_bgr, model
                        )
                        frame_tools_rgb = cv2.cvtColor(frame_tools, cv2.COLOR_BGR2RGB)
                        st.image(frame_tools_rgb, use_container_width=True)
                        
                        if detections:
                            st.success(f"✅ {len(detections)} herramienta(s) encontrada(s)")
                            for i, det in enumerate(detections, 1):
                                st.write(f"{i}. **{det['class']}** - {det['confidence']*100:.1f}%")
                        else:
                            st.info("No se detectaron herramientas")
                
                if detect_persons:
                    with col2:
                        st.subheader("👤 Personas Detectadas")
                        frame_persons, persons = process_pose_detection_static(
                            image_bgr, pose_model
                        )
                        frame_persons_rgb = cv2.cvtColor(frame_persons, cv2.COLOR_BGR2RGB)
                        st.image(frame_persons_rgb, use_container_width=True)
                        
                        if persons:
                            st.success(f"✅ {len(persons)} persona(s) encontrada(s)")
                        else:
                            st.info("No se detectaron personas")
    
    elif mode == "🎥 Cámara en Vivo":
        st.header("🎥 Cámara en Vivo")
        st.success("✅ La velocidad se calcula en tiempo real")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            run = st.checkbox("▶️ Iniciar Cámara", value=False)
        with col2:
            fps_display = st.empty()
        with col3:
            frame_count_display = st.empty()
        
        col_tools, col_persons = st.columns(2)
        with col_tools:
            frame_placeholder_tools = st.empty()
            stats_tools = st.empty()
        with col_persons:
            frame_placeholder_persons = st.empty()
            stats_persons = st.empty()
        
        if run:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            
            prev_time = time.time()
            frame_count = 0
            
            try:
                while run:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ No se puede acceder a la cámara")
                        break
                    
                    frame_count += 1
                    current_time = time.time()
                    time_diff = current_time - prev_time if prev_time > 0 else 0.033
                    fps = 1 / time_diff if time_diff > 0 else 0
                    prev_time = current_time
                    
                    if detect_tools:
                        frame_tools, detections = process_yolo_detection(frame, model)
                        frame_tools_rgb = cv2.cvtColor(frame_tools, cv2.COLOR_BGR2RGB)
                        frame_placeholder_tools.image(
                            frame_tools_rgb,
                            caption="🔧 Herramientas",
                            use_container_width=True
                        )
                        stats_tools.metric("Herramientas", len(detections))
                    
                    if detect_persons:
                        frame_persons, persons = process_pose_detection_live(
                            frame, pose_model, tracker, frame_count, time_diff
                        )
                        frame_persons_rgb = cv2.cvtColor(frame_persons, cv2.COLOR_BGR2RGB)
                        frame_placeholder_persons.image(
                            frame_persons_rgb,
                            caption="👤 Personas + Velocidad",
                            use_container_width=True
                        )
                        stats_persons.metric("Personas", len(persons))
                    
                    fps_display.metric("FPS", f"{fps:.1f}")
                    frame_count_display.metric("Frames", frame_count)
                    
                    time.sleep(0.01)
                    
            finally:
                cap.release()
        else:
            st.info("👆 Activa la casilla para iniciar la cámara")

if __name__ == "__main__":
    main()
