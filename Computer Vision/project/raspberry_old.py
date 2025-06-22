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
    "pothole": 0.5,
    "crack": 0.4,
    "manhole": 0.45  # Puoi modificare o aggiungere altre classi se necessario
}

# Classi di interesse
classes_of_interest = ["pothole", "crack"]

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

while True:
    ret, frame = videoCap.read()
    if not ret:
        continue

    results = model.track(frame, stream=True, imgsz=640)

    # Lista per salvare le righe da scrivere nel file per il frame corrente
    tsv_rows = []

    for result in results:
        # Ottieni i nomi delle classi dal modello
        classes_names = result.names

        # Itera sulle bounding box rilevate nel frame corrente
        for box in result.boxes:
            cls = int(box.cls[0])
            # Converte il nome della classe in minuscolo per uniformità
            class_name = classes_names[cls].lower()
            conf = box.conf[0]
            threshold = confidence_thresholds[class_name]
            
            # Verifica se la confidence supera la soglia impostata per la classe
            if conf < threshold:
                continue

            # Estrae le coordinate della bounding box
            [x1, y1, x2, y2] = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            colour = getColours(cls)

            # Disegna il rettangolo e annota la classe con la confidence
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(frame, f'{class_name} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)

            # Se la detection è di tipo "pothole" o "crack", salva i dati su file TSV
            if class_name in classes_of_interest:
                # Calcola l'area della bounding box in pixel
                area = (x2 - x1) * (y2 - y1)
                # Ottieni il timestamp corrente (locale)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Leggi una riga dalla porta seriale per ottenere i dati GPS
                gps_line = ser.readline().decode('ascii', errors='replace').strip()
                gps_data = parse_nmea(gps_line)
                timestamp_gps = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Timestamp per i dati GPS
                if gps_data:
                    lat, lon = gps_data
                else:
                    lat, lon = (None, None)
                
                # Crea una riga formattata per il file TSV includendo anche la confidence
                tsv_line = f"{timestamp}\t{class_name}\t{area}\t{conf:.2f}\t{timestamp_gps}\t{lat}\t{lon}\n"
                tsv_rows.append(tsv_line)

    # Se ci sono rilevamenti da salvare, appendili al file TSV
    if tsv_rows:
        with open(output_file, "a") as f:
            for line in tsv_rows:
                f.write(line)

    # Visualizza il frame con le annotazioni
    cv2.imshow('frame', frame)

    # Termina il ciclo se viene premuto 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilascia la cattura video e chiude tutte le finestre
videoCap.release()
cv2.destroyAllWindows()

print("Detection terminata.")