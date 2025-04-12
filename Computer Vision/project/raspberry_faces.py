import cv2
from ultralytics import YOLO
import serial
from datetime import datetime
import os

# Carica il modello aggiornato (deve essere addestrato per rilevare "pothole", "crack" e "manhole")
model = YOLO('./best_face.pt')  # Sostituisci con il percorso corretto del tuo modello

# Apre la cattura video (modifica l'indice della telecamera se necessario)
videoCap = cv2.VideoCapture(0)

# Apre la porta seriale per il GPS (modifica il device e il baud rate se necessario)
ser = serial.Serial('/dev/ttyACM0', 4800, timeout=1)

# Nome del file in cui salvare i dati (modalità append)
output_file = "detections.tsv"

# Se il file non esiste, scrivi la riga di header
if not os.path.exists(output_file):
    with open(output_file, "w") as f:
        f.write("timestamp\ttype\tarea\tconfidence\ttimestamp_gps\tlat\tlon\n")
        f.close()

# Dizionario con i valori di confidence per ogni classe
confidence_thresholds = {
    "face": 0.8
}

# Classi di interesse
classes_of_interest = ["face"]

# Funzione per il parsing della stringa NMEA GPRMC
def parse_nmea(sentence):
    """
    Esegue il parsing di una stringa NMEA GPRMC.

    Restituisce una tupla (latitudine, longitudine) se i dati sono validi (stato 'A'),
    altrimenti None.
    """
    try:
        if sentence.startswith('$GPRMC'):
            parts = sentence.split(',')
            if parts[2] == 'A':  # Stato 'A' = dati validi
                lat_raw = parts[3]
                lat_dir = parts[4]
                lon_raw = parts[5]
                lon_dir = parts[6]
                # Conversione da formato ddmm.mmmm (latitudine) e dddmm.mmmm (longitudine) a gradi decimali
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
    except Exception as e:
        return None

# Funzione per ottenere un colore in base al numero della classe
def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] *
             (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

print("Avvio della detection. Premi 'q' per terminare.")

# Variabile per memorizzare l'ultimo dato GPS valido
last_gps_data = None
last_timestamp_gps = None

print("Avvio della detection. Premi 'q' per terminare.")

while True:
    ret, frame = videoCap.read()
    if not ret:
        continue
    
    # Leggi una volta i dati GPS per l'intero frame
    gps_line = ser.readline().decode('ascii', errors='replace').strip()
    gps_data = parse_nmea(gps_line)
    if gps_data:
        last_gps_data = gps_data
        last_timestamp_gps = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    results = model.track(frame, stream=True, imgsz=640)
    tsv_rows = []
    
    for result in results:
        classes_names = result.names
        for box in result.boxes:
            cls = int(box.cls[0])
            class_name = classes_names[cls].lower()
            conf = box.conf[0]
            threshold = confidence_thresholds.get(class_name, 0.4)
            if conf < threshold:
                continue
            
            [x1, y1, x2, y2] = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            colour = getColours(cls)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(frame, f'{class_name} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
            
            # Salva il dato solo se la detection è "pothole" o "crack"
            if class_name in classes_of_interest:
                area = (x2 - x1) * (y2 - y1)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Utilizza l'ultimo dato GPS valido (se disponibile)
                lat, lon = (last_gps_data if last_gps_data else (None, None))
                timestamp_gps = last_timestamp_gps if last_timestamp_gps else timestamp
                tsv_line = f"{timestamp}\t{class_name}\t{area}\t{conf:.2f}\t{timestamp_gps}\t{lat}\t{lon}\n"
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

print("Detection terminata.")