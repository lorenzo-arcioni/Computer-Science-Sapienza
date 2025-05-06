import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import math
import os

# ---------- Configurazione pygame per musica e feedback ----------
pygame.mixer.init()

# Carica playlist dalla cartella ./music
music_folder = "music"
playlist = [os.path.join(music_folder, f) for f in os.listdir(music_folder)
            if f.lower().endswith((".mp3", ".wav", ".ogg"))]
if not playlist:
    raise RuntimeError(f"Nessun file audio trovato in {music_folder}")
current_index = 0
music_paused = False

def load_and_play(index):
    global music_paused
    path = playlist[index]
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    music_paused = False
    print(f"Riproduzione: {os.path.basename(path)}")

# Funzioni di controllo musicale

def play_pause():
    global music_paused
    if pygame.mixer.music.get_busy() and not music_paused:
        pygame.mixer.music.pause()
        music_paused = True
        print("Musica in pausa")
    elif music_paused:
        pygame.mixer.music.unpause()
        music_paused = False
        print("Musica ripresa")
    else:
        load_and_play(current_index)


def next_track():
    global current_index
    current_index = (current_index + 1) % len(playlist)
    load_and_play(current_index)


def prev_track():
    global current_index
    current_index = (current_index - 1) % len(playlist)
    load_and_play(current_index)


def vol_up():
    vol = pygame.mixer.music.get_volume()
    pygame.mixer.music.set_volume(min(1.0, vol + 0.1))
    print(f"Volume: {pygame.mixer.music.get_volume():.1f}")


def vol_dw():
    vol = pygame.mixer.music.get_volume()
    pygame.mixer.music.set_volume(max(0.0, vol - 0.1))
    print(f"Volume: {pygame.mixer.music.get_volume():.1f}")

# Feedback sonori
path_start_sound = os.path.join("feedbacksounds", "start.mp3")
path_completed_sound = os.path.join("feedbacksounds", "stop.mp3")
DETECTION_START_SOUND = pygame.mixer.Sound(path_start_sound)
GESTURE_COMPLETED_SOUND = pygame.mixer.Sound(path_completed_sound)
DETECTION_START_SOUND.set_volume(1)
GESTURE_COMPLETED_SOUND.set_volume(1)

# ---------- Funzioni di riconoscimento gestuale ----------
# (invariato rispetto al tuo codice)

def check_open_hand(hand_landmarks):
    all_y = [lm.y for lm in hand_landmarks.landmark]
    all_x = [lm.x for lm in hand_landmarks.landmark]
    finger_tips = [4, 8, 12, 16, 20]
    finger_pips = [2, 6, 10, 14, 18]
    open_fingers = 0
    if hand_landmarks.landmark[0].y > min(all_y[1:]):
        for i in range(5):
            tip = hand_landmarks.landmark[finger_tips[i]]
            pip = hand_landmarks.landmark[finger_pips[i]]
            if i == 0:
                if (hand_landmarks.landmark[0].x < tip.x and tip.x > pip.x and hand_landmarks.landmark[1].x > hand_landmarks.landmark[0].x) or \
                   (hand_landmarks.landmark[0].x > tip.x and tip.x < pip.x and hand_landmarks.landmark[1].x < hand_landmarks.landmark[0].x):
                    open_fingers += 1
            else:
                if tip.y < pip.y and abs(hand_landmarks.landmark[4].x - hand_landmarks.landmark[20].x) > 0.1 and abs(hand_landmarks.landmark[0].y - hand_landmarks.landmark[12].y) > 0.3:
                    open_fingers += 1
        return open_fingers >= 5
    return False


def check_thumbs_up(hand_landmarks):
    all_y = [lm.y for lm in hand_landmarks.landmark]
    thumb_y = all_y[3:5]
    other_y = all_y[5:]
    return max(thumb_y) < min(other_y) and abs(hand_landmarks.landmark[4].y - hand_landmarks.landmark[5].y) > 0.1


def check_thumbs_down(hand_landmarks):
    all_y = [lm.y for lm in hand_landmarks.landmark]
    thumb_y = all_y[3:5]
    other_y = all_y[5:]
    return (min(thumb_y) > max(other_y) and abs(hand_landmarks.landmark[4].x - hand_landmarks.landmark[17].x) < 0.2 and
            abs(hand_landmarks.landmark[4].x - hand_landmarks.landmark[9].x) < 0.2)


def check_thumbs_rx(hand_landmarks):
    all_y = [lm.y for lm in hand_landmarks.landmark]
    all_x = [lm.x for lm in hand_landmarks.landmark]
    thumb_x = all_x[2:5]
    other_x = all_x[5:]
    ind_n = [5, 9, 13, 17]
    ind_t = [8, 12, 16, 20]
    nocche = [all_y[i] for i in ind_n]
    tips = [all_y[i] for i in ind_t]
    dita_chiuse = all(all_y[t] > all_y[n] for t, n in zip(ind_t, ind_n))
    return (min(thumb_x) > max(other_x) and abs(hand_landmarks.landmark[4].y - hand_landmarks.landmark[17].y) < 0.2 and
            abs(hand_landmarks.landmark[4].y - hand_landmarks.landmark[9].y) < 0.2 )#and min(tips) > max(nocche) and dita_chiuse)


def check_thumbs_sx(hand_landmarks):
    all_y = [lm.y for lm in hand_landmarks.landmark]
    all_x = [lm.x for lm in hand_landmarks.landmark]
    thumb_x = all_x[2:5]
    other_x = all_x[5:]
    ind_n = [5, 9, 13, 17]
    ind_t = [8, 12, 16, 20]
    nocche = [all_y[i] for i in ind_n]
    tips = [all_y[i] for i in ind_t]
    dita_chiuse = all(all_y[t] > all_y[n] for t, n in zip(ind_t, ind_n))
    return (min(thumb_x) < max(other_x) and abs(hand_landmarks.landmark[4].y - hand_landmarks.landmark[17].y) < 0.2 and
            abs(hand_landmarks.landmark[4].y - hand_landmarks.landmark[9].y) < 0.2 and dita_chiuse)


def fuuu(hand_landmarks):
    all_y = [lm.y for lm in hand_landmarks.landmark]
    combined = all_y[0:8] + all_y[13:]
    return hand_landmarks.landmark[11].y < min(combined)

# Variabili globali per il rilevamento
last_trigger_time = 0
cooldown = 1
is_detecting = False
detection_start_time = 0
detection_delay = 0.9
current_gesture = None
consecutive_thumbs_up_count = 0
consecutive_thumbs_down_count = 0
last_gesture = None

# Funzione di gestione gesti

def handle_gesture(name, action_fn, hand_landmarks, frame):
    global last_trigger_time, detection_start_time, is_detecting, current_gesture
    global consecutive_thumbs_up_count, consecutive_thumbs_down_count, last_gesture
    if not is_detecting:
        is_detecting = True
        detection_start_time = time.time()
        current_gesture = name
        DETECTION_START_SOUND.play()
        return
    if current_gesture != name:
        detection_start_time = time.time()
        current_gesture = name
        DETECTION_START_SOUND.play()
        return
    elapsed = time.time() - detection_start_time
    x, y = int(hand_landmarks.landmark[0].x * frame.shape[1]), int(hand_landmarks.landmark[0].y * frame.shape[0])
    cv2.putText(frame, name.upper(), (x, y-70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    center = (x, y-20)
    radius = 30
    cv2.circle(frame, center, radius, (100,100,100), 3)
    start_ang = -90
    end_ang = start_ang + min(elapsed/detection_delay*360, 360)
    for i in range(int(start_ang), int(end_ang)):
        rad = math.radians(i)
        cv2.circle(frame, (int(center[0]+radius*math.cos(rad)), int(center[1]+radius*math.sin(rad))), 1, (0,255,0), 3)
    if elapsed >= detection_delay:
        action_fn()
        print(f"{name.upper()} rilevato!")
        GESTURE_COMPLETED_SOUND.play()
        if name == "thumbs_up":
            if last_gesture == name: consecutive_thumbs_up_count += 1
            else: consecutive_thumbs_up_count = 1
            consecutive_thumbs_down_count = 0
        elif name == "thumbs_down":
            if last_gesture == name: consecutive_thumbs_down_count += 1
            else: consecutive_thumbs_down_count = 1
            consecutive_thumbs_up_count = 0
        else:
            consecutive_thumbs_up_count = consecutive_thumbs_down_count = 0
        last_gesture = name
        last_trigger_time = time.time()
        is_detecting = False
        current_gesture = None

def estimate_hand_side(landmarks, handedness):
    WRIST = 0
    INDEX_MCP = 5
    PINKY_MCP = 17

    wrist = np.array([landmarks[WRIST].x, landmarks[WRIST].y, landmarks[WRIST].z])
    index_mcp = np.array([landmarks[INDEX_MCP].x, landmarks[INDEX_MCP].y, landmarks[INDEX_MCP].z])
    pinky_mcp = np.array([landmarks[PINKY_MCP].x, landmarks[PINKY_MCP].y, landmarks[PINKY_MCP].z])

    vec1 = index_mcp - wrist
    vec2 = pinky_mcp - wrist

    normal = np.cross(vec1, vec2)

    # Se la mano è sinistra, invertiamo la normale
    if handedness == 'Left':
        normal = -normal

    if normal[2] > 0:
        return "PALM"
    else:
        return "BACK"

# Impostazione MediaPipe e videocamera
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(2)

# Avvia la riproduzione iniziale
load_and_play(current_index)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if is_detecting and (time.time() - detection_start_time > detection_delay * 1.5):
        is_detecting = False
        current_gesture = None
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            skip_cd = False
            if last_gesture == "thumbs_up" and consecutive_thumbs_up_count >= 2 and check_thumbs_up(hand_landmarks): skip_cd = True
            if last_gesture == "thumbs_down" and consecutive_thumbs_down_count >= 2 and check_thumbs_down(hand_landmarks): skip_cd = True
            if ((time.time() - last_trigger_time) > cooldown) or skip_cd:
                if check_open_hand(hand_landmarks) and estimate_hand_side(hand_landmarks.landmark, handedness.classification[0].label) == "PALM":
                    handle_gesture("open_hand", play_pause, hand_landmarks, frame)
                elif check_thumbs_up(hand_landmarks):
                    handle_gesture("thumbs_up", vol_up, hand_landmarks, frame)
                elif check_thumbs_rx(hand_landmarks):
                    handle_gesture("thumbs_rx", next_track, hand_landmarks, frame)
                elif check_thumbs_sx(hand_landmarks):
                    handle_gesture("thumbs_sx", prev_track, hand_landmarks, frame)
                elif check_thumbs_down(hand_landmarks):
                    handle_gesture("thumbs_down", vol_dw, hand_landmarks, frame)
    if consecutive_thumbs_up_count >= 2:
        cv2.putText(frame, f"Thumbs UP mode: {consecutive_thumbs_up_count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    if consecutive_thumbs_down_count >= 2:
        cv2.putText(frame, f"Thumbs DOWN mode: {consecutive_thumbs_down_count}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.imshow("Gesture Music Player", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
