import cv2
import mediapipe as mp
import pygame
import os
import time
import numpy as np

# ==========================
# Configurazione Pygame
# ==========================
pygame.mixer.init()

music_folder = "songs"  # cartella con i file MP3
playlist = [os.path.join(music_folder, f) for f in os.listdir(music_folder) if f.endswith(".mp3")]
if not playlist:
    raise Exception("Non sono stati trovati file .mp3 nella cartella 'music'.")

current_track = 0
pygame.mixer.music.load(playlist[current_track])
pygame.mixer.music.set_volume(0.5)  # volume iniziale al 50%
pygame.mixer.music.play()
music_paused = False  # stato della riproduzione

# ==========================
# Variabili di Debounce
# ==========================
debounce_time = {
    "swipe": 1.0,    # tempo minimo tra swipe
    "pause": 1.0
    # Il volume è aggiornato in tempo reale, perciò non viene usato il debouncing.
}

last_action = {
    "swipe": 0,
    "pause": 0
}

# Per lo swipe memorizziamo la posizione orizzontale media della mano
prev_center_x = None

# ==========================
# Funzioni di Controllo
# ==========================
def pause_music():
    global music_paused
    now = time.time()
    if now - last_action["pause"] < debounce_time["pause"]:
        return
    if not music_paused:
        pygame.mixer.music.pause()
        music_paused = True
        print("Pausa")
    last_action["pause"] = now

def resume_music():
    global music_paused
    now = time.time()
    if now - last_action["pause"] < debounce_time["pause"]:
        return
    if music_paused:
        pygame.mixer.music.unpause()
        music_paused = False
        print("Riprendi")
    last_action["pause"] = now

def next_track():
    global current_track, music_paused
    now = time.time()
    if now - last_action["swipe"] < debounce_time["swipe"]:
        return
    current_track = (current_track + 1) % len(playlist)
    pygame.mixer.music.load(playlist[current_track])
    pygame.mixer.music.play()
    music_paused = False
    print("Brano Successivo:", os.path.basename(playlist[current_track]))
    last_action["swipe"] = now

def previous_track():
    global current_track, music_paused
    now = time.time()
    if now - last_action["swipe"] < debounce_time["swipe"]:
        return
    current_track = (current_track - 1) % len(playlist)
    pygame.mixer.music.load(playlist[current_track])
    pygame.mixer.music.play()
    music_paused = False
    print("Brano Precedente:", os.path.basename(playlist[current_track]))
    last_action["swipe"] = now

# ==========================
# Funzione di rilevamento dello stato delle dita
# Per index, middle, ring e pinky: se il tip (landmark 8, 12, 16, 20) ha coordinate y minori rispetto al rispettivo PIP (6, 10, 14, 18) si considera la dito esteso.
# Per il pollice si utilizza una comparazione orizzontale (tip > CMC), dato l'effetto del flip.
# ==========================
def fingers_status(hand_landmarks):
    finger = {}
    finger['index'] = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
    finger['middle'] = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
    finger['ring'] = hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y
    finger['pinky'] = hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y
    finger['thumb'] = hand_landmarks.landmark[4].x > hand_landmarks.landmark[1].x
    return finger

# ==========================
# Configurazione MediaPipe e OpenCV
# ==========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, 
                       min_detection_confidence=0.7, 
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inverte l'immagine per effetto specchio e converte BGR -> RGB.
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    gesture_triggered = False  # flag per evitare conflitti fra gesti
    height, width, _ = frame.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Disegna i landmark della mano
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Calcola il centro della mano (media dei landmark)
            coords = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
            center_x = np.mean(coords[:, 0])
            center_y = np.mean(coords[:, 1])

            # Estrarre lo stato delle dita
            status = fingers_status(hand_landmarks)

            # ---------------------------
            # 1. Cambio Brano: Swipe orizzontale della mano
            # Se il centro della mano si sposta sufficientemente a destra o sinistra rispetto al frame precedente
            # ---------------------------
            swipe_threshold = 0.12  # soglia empirica
            if prev_center_x is not None:
                delta_x = center_x - prev_center_x
                if delta_x > swipe_threshold:
                    next_track()
                    gesture_triggered = True
                elif delta_x < -swipe_threshold:
                    previous_track()
                    gesture_triggered = True
            prev_center_x = center_x  # aggiorna per il frame successivo

            # Se uno swipe è stato attivato, non eseguo altri gesti in questo ciclo
            if gesture_triggered:
                continue

            # ---------------------------
            # 2. Pausa / Riprendi:
            #    Se il palmo è aperto (tutte le dita tranne il pollice estese) → pausa
            #    Se viene formato un pugno (nessuna delle dita estese) → riprendi
            # ---------------------------
            # Calcolo quante dita (index, middle, ring, pinky) sono estese
            fingers_extended = sum([status['index'], status['middle'], status['ring'], status['pinky']])
            if fingers_extended == 4:
                pause_music()
                gesture_triggered = True
            elif fingers_extended == 0:
                resume_music()
                gesture_triggered = True

            if gesture_triggered:
                continue

            # ---------------------------
            # 3. Controllo Volume:
            #    Utilizza la distanza tra pollice (landmark 4) e indice (landmark 8).
            #    Mappa la distanza in un range volume: quando le dita si avvicinano il volume diminuisce,
            #    quando si allontanano aumenta. Viene inoltre disegnata la linea che collega i due punti.
            # ---------------------------
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            # Converti coordinate normalizzate in pixel
            thumb_pos = (int(thumb_tip.x * width), int(thumb_tip.y * height))
            index_pos = (int(index_tip.x * width), int(index_tip.y * height))
            # Disegna la linea che unisce pollice e indice
            cv2.line(frame, thumb_pos, index_pos, (0, 255, 0), 3)
            # Disegna i cerchi alle estremità
            cv2.circle(frame, thumb_pos, 5, (0, 0, 255), -1)
            cv2.circle(frame, index_pos, 5, (0, 0, 255), -1)

            # Calcola la distanza euclidea (usando le coordinate normalizzate)
            distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5

            # Parametri di calibrazione: questi valori possono essere regolati
            min_distance = 0.02  # distanza minima rilevata (volume minimo)
            max_distance = 0.2   # distanza massima rilevata (volume massimo)

            # Mappa la distanza a un valore volume compreso tra 0 e 1
            vol = (distance - min_distance) / (max_distance - min_distance)
            vol = np.clip(vol, 0, 1)
            pygame.mixer.music.set_volume(vol)
            # Visualizza il volume corrente sul frame (in percentuale)
            cv2.putText(frame, f'Volume: {int(vol*100)}%', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Controllo Musica con Gesti', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()