import cv2
from ultralytics import YOLO
import time
import numpy as np
import threading
from queue import Queue
from threading import Lock, Event
from collections import deque

# ============================================
# CONFIGURACIÓN
# ============================================
MODEL_PATH = "runs/detect/train4/weights/best.pt"
model = YOLO(MODEL_PATH)
pose_model = YOLO('yolov8n-pose.pt')

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
# SINCRONIZACIÓN
# ============================================
frame_lock = Lock()
results_lock = Lock()
stop_event = Event()

frame_buffer = deque(maxlen=3)
frame_buffer_lock = Lock()

last_yolo_results = {'boxes': [], 'frame_num': 0}
last_pose_results = {'detections': [], 'frame_num': 0}

# ============================================
# TRACKER + SUAVIZADO REAL DE VELOCIDAD
# ============================================
class PersonTracker:
    def __init__(self):
        self.persons = {}
        self.next_id = 1
        self.max_distance = 150
        self.lock = Lock()
        self.alpha = 0.2  # suavizado EMA

    def update(self, center, frame_number, time_diff):
        with self.lock:
            min_dist = float('inf')
            matched_id = None
            
            # Buscar persona previa
            for pid, data in self.persons.items():
                if frame_number - data['last_seen'] < 30:
                    prev_x, prev_y = data['position']
                    dist = np.sqrt((center[0]-prev_x)**2 + (center[1]-prev_y)**2)
                    if dist < min_dist and dist < self.max_distance:
                        min_dist = dist
                        matched_id = pid
            
            # Si coincide con alguien → actualizar
            if matched_id:
                prev_vel_raw = self.persons[matched_id]['velocity_raw']
                vel_raw = min_dist / time_diff if time_diff > 0 else 0
                
                # SUAVIZADO EMA
                vel_suave = (self.alpha * vel_raw) + ((1 - self.alpha) * prev_vel_raw)

                # Normalización a escala 0–10
                vel_norm = min(10, vel_suave / 25)

                self.persons[matched_id] = {
                    'position': center,
                    'velocity_raw': vel_suave,
                    'velocity_norm': vel_norm,
                    'last_seen': frame_number
                }
                return matched_id, vel_norm
            
            # Nueva persona
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
        with self.lock:
            eliminar = [pid for pid, data in self.persons.items()
                        if frame_number - data['last_seen'] > max_frames]
            for pid in eliminar:
                del self.persons[pid]

tracker = PersonTracker()

# ============================================
# UTILIDADES
# ============================================
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
                         (255, 100, 50), 4)

    for (x, y) in keypoints:
        if x > 0 and y > 0:
            cv2.circle(frame, (int(x), int(y)), 5, (50, 255, 255), -1)


def get_velocity_color(vel_norm):
    if vel_norm < 3:
        return (50, 255, 50), "LENTO"
    elif vel_norm < 6:
        return (50, 255, 255), "NORMAL"
    else:
        return (50, 50, 255), "RÁPIDO"


def draw_person_info(frame, cx, cy, pid, vel):

    color, status = get_velocity_color(vel)

    # Punto central
    cv2.circle(frame, (cx, cy), 18, (255, 255, 255), -1)
    cv2.circle(frame, (cx, cy), 15, color, -1)
    cv2.circle(frame, (cx, cy), 18, color, 3)

    # Panel
    panel_x = cx - 100
    panel_y = cy - 115
    w, h = 200, 95

    cv2.rectangle(frame, (panel_x - 3, panel_y - 3),
                  (panel_x + w + 3, panel_y + h + 3), (0, 0, 0), -1)
    cv2.rectangle(frame, (panel_x, panel_y),
                  (panel_x + w, panel_y + h), (40, 40, 40), -1)
    cv2.rectangle(frame, (panel_x, panel_y),
                  (panel_x + w, panel_y + h), color, 4)

    # Header
    cv2.rectangle(frame, (panel_x, panel_y),
                  (panel_x + w, panel_y + 30), (60, 60, 60), -1)

    cv2.putText(frame, f"PERSONA #{pid}", (panel_x + 20, panel_y + 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)

    # Velocidad normalizada
    cv2.putText(frame, f"Velocidad: {vel:.1f}/10",
                (panel_x + 20, panel_y + 60),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

    cv2.putText(frame, status, (panel_x + 20, panel_y + 88),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 2)


# ============================================
# HILO YOLO
# ============================================
def yolo_thread_func():
    global last_yolo_results
    while not stop_event.is_set():
        try:
            with frame_buffer_lock:
                if len(frame_buffer) == 0:
                    continue
                fd = frame_buffer[-1]

            frame = fd['frame']
            frame_num = fd['frame_num']

            results = model(frame, conf=0.5, verbose=False)

            boxes = []
            for r in results:
                for b in r.boxes:
                    boxes.append({
                        'xyxy': b.xyxy[0].cpu().numpy(),
                        'cls': int(b.cls[0]),
                        'conf': float(b.conf[0])
                    })

            with results_lock:
                last_yolo_results = {
                    'boxes': boxes,
                    'frame_num': frame_num
                }

            time.sleep(0.005)

        except Exception as e:
            print("YOLO ERROR:", e)


# ============================================
# HILO POSE
# ============================================
def pose_thread_func():
    global last_pose_results
    while not stop_event.is_set():
        try:
            with frame_buffer_lock:
                if len(frame_buffer) == 0:
                    continue
                fd = frame_buffer[-1]

            frame = fd['frame']
            frame_num = fd['frame_num']
            time_diff = fd['time_diff']

            results = pose_model(frame, conf=0.5, verbose=False)

            persons = []
            for r in results:
                if r.keypoints is not None:
                    kps = r.keypoints.xy
                    for i in range(kps.shape[0]):
                        persons.append({'keypoints': kps[i].cpu().numpy()})

            with results_lock:
                last_pose_results = {
                    'detections': persons,
                    'frame_num': frame_num,
                    'time_diff': time_diff
                }

            time.sleep(0.005)

        except Exception as e:
            print("POSE ERROR:", e)


# ============================================
# MAIN
# ============================================
def main():
    frame_count = 0
    prev_time = time.time()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se puede abrir cámara")
        return

    cap.set(3, 960)
    cap.set(4, 540)

    cv2.namedWindow("PROYECTO FINAL", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("PROYECTO FINAL", 1920, 540)

    threading.Thread(target=yolo_thread_func, daemon=True).start()
    threading.Thread(target=pose_thread_func, daemon=True).start()

    while not stop_event.is_set():

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        now = time.time()
        time_diff = now - prev_time
        prev_time = now

        height, width = frame.shape[:2]

        with frame_buffer_lock:
            frame_buffer.append({
                'frame': frame.copy(),
                'frame_num': frame_count,
                'time_diff': time_diff
            })

        # COPIAS
        frame_yolo = frame.copy()
        frame_pose = frame.copy()

        # ===============================================
        # YOLO DRAW
        # ===============================================
        with results_lock:
            ydata = last_yolo_results.copy()

        tools_count = 0

        for box in ydata['boxes']:
            tools_count += 1

            x1, y1, x2, y2 = box['xyxy'].astype(int)
            cls = box['cls']
            conf = box['conf']

            # Reducir cuadro al 75%
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            w = int((x2 - x1) * 0.75)
            h = int((y2 - y1) * 0.75)

            x1 = max(0, cx - w // 2)
            y1 = max(70, cy - h // 2)
            x2 = min(width - 1, cx + w // 2)
            y2 = min(height - 60, cy + h // 2)

            cv2.rectangle(frame_yolo, (x1, y1), (x2, y2), (0, 255, 0), 4)

            label = f"{CLASS_NAMES[cls]} ({conf*100:.0f}%)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame_yolo, (x1, y1 - th - 10), (x1 + tw + 5, y1), (0, 0, 0), -1)
            cv2.putText(frame_yolo, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # HEADER
        cv2.rectangle(frame_yolo, (0, 0), (width, 60), (30, 30, 30), -1)
        cv2.putText(frame_yolo, "CLASIFICACION DE HERRAMIENTAS",
                    (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # FOOTER
        cv2.rectangle(frame_yolo, (0, height - 50), (width, height), (30, 30, 30), -1)
        cv2.putText(frame_yolo, f"Detecciones: {tools_count}",
                    (15, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

        # ===============================================
        # POSE DRAW
        # ===============================================
        with results_lock:
            pdata = last_pose_results.copy()

        persons_detected = 0

        for person in pdata['detections']:
            persons_detected += 1

            kp = person['keypoints']
            draw_pose_skeleton(frame_pose, kp)

            cx, cy = calculate_center_pose(kp, width, height)
            pid, vel_norm = tracker.update((cx, cy), frame_count, time_diff)
            draw_person_info(frame_pose, cx, cy, pid, vel_norm)

        tracker.cleanup(frame_count)

        # Header pose
        cv2.rectangle(frame_pose, (0, 0), (width, 60), (30, 30, 30), -1)
        cv2.putText(frame_pose, "PERSONAS + VELOCIDAD",
                    (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Footer pose
        cv2.rectangle(frame_pose, (0, height - 50), (width, height), (30, 30, 30), -1)
        cv2.putText(frame_pose, f"Personas: {persons_detected} | IDs: {tracker.next_id - 1}",
                    (15, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

        # COMBINAR LADO A LADO
        combined = np.hstack((frame_yolo, frame_pose))
        cv2.line(combined, (width, 0), (width, height), (200, 200, 200), 2)

        cv2.imshow("PROYECTO FINAL", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            stop_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()
    stop_event.set()


if __name__ == "__main__":
    main()
