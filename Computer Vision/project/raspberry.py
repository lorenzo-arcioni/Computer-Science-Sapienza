import cv2
from ultralytics import YOLO
import serial
from datetime import datetime
import os
import time  # Per calcolare il tempo

model = YOLO('./best_face.pt')
videoCap = cv2.VideoCapture(0)
ser = serial.Serial('/dev/ttyACM0', 4800, timeout=1)
output_file = "detections-faces.tsv"

# Intestazione TSV con nuova colonna
if not os.path.exists(output_file):
    with open(output_file, "w") as f:
        f.write("timestamp\ttype\tarea\tconfidence\ttimestamp_gps\tlat\tlon\tinference_time_ms\n")

confidence_thresholds = {
    "pothole": 0.5,
    "crack": 0.4,
    "manhole": 0.45,
    "face": 0.5
}

classes_of_interest = ['face']

def parse_nmea(sentence):
    try:
        if sentence.startswith('$GPRMC'):
            parts = sentence.split(',')
            if parts[2] == 'A':
                lat_raw = parts[3]
                lat_dir = parts[4]
                lon_raw = parts[5]
                lon_dir = parts[6]
                lat_deg = float(lat_raw[:2])
                lat_min = float(lat_raw[2:])
                lat = lat_deg + lat_min / 60.0
                if lat_dir == 'S':
                    lat = -lat
                lon_deg = float(lon_raw[:3])
                lon_min = float(lon_raw[3:])
                lon = lon_deg + lon_min / 60.0
                if lon_dir == 'W':
                    lon = -lon
                return (lat, lon)
        return None
    except Exception:
        return None

def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] *
             (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

print("Avvio della detection. Premi 'q' per terminare.")

saved_ids = set()

while True:
    ret, frame = videoCap.read()
    if not ret:
        continue

    # ⏱ Calcola tempo di inferenza
    start_time = time.time()
    results = model.track(frame, stream=True, imgsz=640)
    inference_time_ms = round((time.time() - start_time) * 1000)  # in millisecondi

    tsv_rows = []

    for result in results:
        classes_names = result.names
        for box in result.boxes:
            cls = int(box.cls[0])
            class_name = classes_names[cls].lower()
            conf = box.conf[0]
            threshold = confidence_thresholds.get(class_name, 1.0)

            if conf < threshold:
                continue

            [x1, y1, x2, y2] = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            colour = getColours(cls)

            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(frame, f'{class_name} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)

            if class_name in classes_of_interest:
                detection_id = None
                if hasattr(box, 'id') and box.id is not None:
                    detection_id = box.id[0].item()

                if detection_id is None:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    detection_id = hash((x1, y1, x2, y2, timestamp))

                if detection_id not in saved_ids:
                    saved_ids.add(detection_id)

                    area = (x2 - x1) * (y2 - y1)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    gps_line = ser.readline().decode('ascii', errors='replace').strip()
                    gps_data = parse_nmea(gps_line)
                    timestamp_gps = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    lat, lon = gps_data if gps_data else (None, None)

                    # 📝 Riga con tempo di inferenza
                    tsv_line = f"{timestamp}\t{class_name}\t{area}\t{conf:.2f}\t{timestamp_gps}\t{lat}\t{lon}\t{inference_time_ms}\n"
                    tsv_rows.append(tsv_line)

    if tsv_rows:
        with open(output_file, "a") as f:
            for line in tsv_rows:
                f.write(line)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCap.release()
cv2.destroyAllWindows()
ser.close()
print("Detection terminata.")

# Chiudi la porta seriale
ser.close()