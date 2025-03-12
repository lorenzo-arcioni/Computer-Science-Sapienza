# video_processing.py
import os
import cv2
import random
import json
import subprocess
import math
from tqdm import tqdm
from ultralytics import YOLO

def create_temp_video(input_video_path, temp_video_path, yolo_width=640, yolo_height=384):
    """
    Crea un video temporaneo scalato (ad esempio, 640x384) a partire dal video originale.
    Restituisce fps, numero totale di frame e le dimensioni originali.
    """
    cap_in = cv2.VideoCapture(input_video_path)
    if not cap_in.isOpened():
        raise Exception(f"Impossibile aprire il video di input: {input_video_path}")
    
    orig_width  = int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps         = cap_in.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_in.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Scrittore per il video temporaneo (usa codec 'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (yolo_width, yolo_height))
    
    pbar = tqdm(total=total_frames, desc="Scaling video frames", unit="frame")
    while True:
        ret, frame = cap_in.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (yolo_width, yolo_height))
        temp_writer.write(frame_resized)
        pbar.update(1)
    pbar.close()
    
    cap_in.release()
    temp_writer.release()
    return fps, total_frames, (orig_width, orig_height)

def run_yolo_predictions(temp_video_path, model, confidence_threshold, fps, total_frames, prediction_fps=15, skip_factor=None, class_filter=None, position_filter=None):
    """
    Esegue le predizioni YOLO sul video temporaneo.
    Utilizza il parametro 'vid_stride' per ridurre il numero di frame elaborati e
    applica dei filtri per le confidence a seconda della classe:
      - manhole (classe 2): confidenza >= 80%
      - pothole (classe 0): confidenza >= 20%
      - crack (classe 1): confidenza >= 40%
    Restituisce una lista di bounding box per ogni frame.
    
    Il parametro prediction_fps permette di specificare quanti fps usare per le predizioni.
    """
    if skip_factor is None:
        skip_factor = max(1, round(fps / prediction_fps))
    print(f"Input fps: {fps:.2f}. Utilizzo vid_stride={skip_factor} per predire ~{prediction_fps} fps.")
    
    processed_results = {}
    total_predictions = math.ceil(total_frames / skip_factor)
    # Progress bar per le predizioni YOLO; verbose=False per non stampare messaggi interni
    for i, result in enumerate(tqdm(model.predict(
            source=temp_video_path,
            save=False,
            stream=True,
            vid_stride=skip_factor,
            conf=confidence_threshold,
            verbose=False), 
            total=total_predictions, desc="YOLO Predictions", unit="frame")):
        processed_frame_idx = i * skip_factor
        if result.boxes is not None and result.boxes.xywh is not None:
            xywh = result.boxes.xywh.cpu().numpy()  # (x_center, y_center, width, height)
            cls  = result.boxes.cls.cpu().numpy()    # indice della classe
            conf = result.boxes.conf.cpu().numpy()    # confidenza
            boxes_data = []
            for j in range(len(xywh)):
                # Applica i filtri in base alla classe e alla posizione
                if not(class_filter is None or class_filter(cls[j], conf[j])):
                    continue
                if not(position_filter is None or position_filter(xywh[j])):
                    continue

                boxes_data.append({
                'xywh': xywh[j],
                'cls': int(cls[j]),
                'conf': float(conf[j])
                })
        else:
            boxes_data = None
        processed_results[processed_frame_idx] = boxes_data

    # Propaga le predizioni ai frame non processati: per ogni frame usa le BB del frame processato precedente
    boxes_per_frame = []
    processed_indices = sorted(processed_results.keys())
    current_pointer = 0
    last_boxes = processed_results.get(processed_indices[0], None)
    for i in tqdm(range(total_frames), desc="Propagating predictions", unit="frame"):
        if current_pointer < len(processed_indices) and i == processed_indices[current_pointer]:
            last_boxes = processed_results[processed_indices[current_pointer]]
            current_pointer += 1
        boxes_per_frame.append(last_boxes)
    
    return boxes_per_frame

def annotate_video(input_video_path, temp_video_path, output_video_path, boxes_per_frame, model_names, yolo_width=640, yolo_height=384):
    """
    Disegna le bounding box (e le relative etichette) sui frame originali e salva il video annotato.
    """
    cap_orig = cv2.VideoCapture(input_video_path)
    if not cap_orig.isOpened():
        raise Exception(f"Impossibile riaprire il video di input: {input_video_path}")
    
    orig_width  = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps         = cap_orig.get(cv2.CAP_PROP_FPS)
    fourcc      = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer  = cv2.VideoWriter(temp_video_path, fourcc, fps, (orig_width, orig_height))
    
    scale_x = orig_width / yolo_width
    scale_y = orig_height / yolo_height
    
    random.seed(42)
    colors = {cls: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
              for cls in model_names.keys()}
    
    total_frames_orig = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    pbar = tqdm(total=total_frames_orig, desc="Annotating video frames", unit="frame")
    while True:
        ret, frame = cap_orig.read()
        if not ret:
            break
        boxes = boxes_per_frame[frame_idx]
        if boxes is not None:
            for box in boxes:
                x_center, y_center, w, h = box['xywh']
                class_id = box['cls']
                confidence = box['conf']
                x_center *= scale_x
                y_center *= scale_y
                w *= scale_x
                h *= scale_y
                top_left = (int(x_center - w/2), int(y_center - h/2))
                bottom_right = (int(x_center + w/2), int(y_center + h/2))
                color = colors.get(class_id, (0, 255, 0))
                cv2.rectangle(frame, top_left, bottom_right, color, 3)
                
                class_name = model_names.get(class_id, "Unknown")
                label = f"{class_name} {confidence * 100:.1f}%"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                text_thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
                text_offset_x = top_left[0]
                text_offset_y = top_left[1] - 10
                if text_offset_y - text_height - baseline < 0:
                    text_offset_y = top_left[1] + text_height + baseline + 10
                box_coords = ((text_offset_x, text_offset_y - text_height - baseline - 4),
                              (text_offset_x + text_width + 4, text_offset_y))
                cv2.rectangle(frame, box_coords[0], box_coords[1], color, cv2.FILLED)
                cv2.putText(frame, label, (text_offset_x + 2, text_offset_y - 2),
                            font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)
        out_writer.write(frame)
        frame_idx += 1
        pbar.update(1)
    pbar.close()
    
    cap_orig.release()
    out_writer.release()

def get_input_bitrate(video_path):
    """
    Estrae il bitrate del video di input utilizzando ffprobe.
    """
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=bit_rate', '-of', 'json', video_path
    ]
    output = subprocess.check_output(cmd)
    data = json.loads(output)
    bitrate = int(data['streams'][0]['bit_rate'])
    return bitrate

def reencode_video(temp_video_path, output_video_path, input_video_path):
    """
    Ri-encoda il video intermedio con FFmpeg mantenendo il bitrate del video originale.
    """
    input_bitrate = get_input_bitrate(input_video_path)
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-i', temp_video_path,
        '-c:v', 'libx264', '-b:v', str(input_bitrate),
        output_video_path
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    try:
        os.remove(temp_video_path)
    except Exception as e:
        print(f"Impossibile rimuovere il file temporaneo: {e}")

def process_video(sweep, input_video_path, output_video_path, confidence_threshold=0.20, prediction_fps=15, do_reencode=True, class_filter=None, position_filter=None):
    """
    Funzione principale per processare un video.
    Carica il modello corrispondente allo sweep, crea il video temporaneo scalato,
    esegue le predizioni YOLO (utilizzando prediction_fps per il calcolo dello skip),
    annota il video e (opzionalmente) ri-encoda l'output.
    """
    model = YOLO(sweep)
    
    file_dir, file_name = os.path.split(input_video_path)
    temp_video_path = os.path.join(file_dir, f"temp_scaled_{file_name}")
    
    fps, total_frames, orig_dims = create_temp_video(input_video_path, temp_video_path)
    print("Video temporaneo scalato creato.")
    
    boxes_per_frame = run_yolo_predictions(temp_video_path, model, confidence_threshold, fps, total_frames, prediction_fps=prediction_fps, class_filter=class_filter, position_filter=position_filter)
    print("Predizioni eseguite sul video scalato.")
    
    annotate_video(input_video_path, temp_video_path, temp_video_path, boxes_per_frame, model.names)
    print(f"Video annotato salvato temporaneamente come {temp_video_path}")
    
    if do_reencode:
        reencode_video(temp_video_path, output_video_path, input_video_path)
        print(f"Video finale salvato come {output_video_path}")
    else:
        os.rename(temp_video_path, output_video_path)
        print(f"Video finale salvato come {output_video_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Elabora un video usando YOLO e annota le predizioni.")
    parser.add_argument("--sweep", type=str, required=True, help="Path dello sweep")
    parser.add_argument("--input_video", type=str, required=True, help="Percorso del video di input")
    parser.add_argument("--output_video", type=str, required=True, help="Percorso del video di output")
    parser.add_argument("--confidence", type=float, default=0.20, help="Soglia di confidenza")
    parser.add_argument("--prediction_fps", type=float, default=15, help="Numero di fps per le predizioni")
    parser.add_argument("--reencode", action="store_true", help="Flag per ri-encodare il video")
    parser.add_argument("--class_filter", type=function, default=None, help="Funzione per filtrare le classi")
    args = parser.parse_args()
    process_video(args.sweep, args.input_video, args.output_video, args.confidence, args.prediction_fps, args.reencode, args.class_filter)
