#import

import cv2
import mediapipe as mp
import time
import math
import os
import pygame
import numpy as np

# pygame player

pygame.init()
pygame.mixer.init()

#se la musica è in pausa
is_paused=False

# Specifica il percorso della cartella
music_folder = './music'

#salva la musica nella cartella
playlist = [f for f in os.listdir(music_folder) if f.endswith((".mp3", ".wav"))]*3

# Indice del brano corrente
current_song_index = 0

# Volume iniziale (da 0.0 a 1.0)
volume = 0.5

def load_song(index):
    global volume
    if 0 <= index < len(playlist):
        song_path = os.path.join(music_folder, playlist[index])
        pygame.mixer.music.load(song_path)
        pygame.mixer.music.set_volume(volume)
        pygame.mixer.music.play()
        print(f"🎵 Riproduzione: {playlist[index]}")
    else:
        print("❌ Indice canzone fuori range!")

# Carica la prima canzone
load_song(current_song_index)
        
def play_pause():
    global is_paused

    if is_paused:
            pygame.mixer.music.unpause()
            is_paused = False
            print("▶️ Ripresa")
    else:
            pygame.mixer.music.pause()
            is_paused = True
            print("⏸️ Pausa")
    


def unpause():
    pygame.mixer.music.unpause()
    print("▶️ Ripresa")

# def stop():
#     pygame.mixer.music.stop()
#     print("⏹️ Stop")

def next_song():
    global is_paused
    global current_song_index
    current_song_index = (current_song_index + 1) % len(playlist)
    print(current_song_index)
    print((current_song_index + 1) % len(playlist))
    load_song(current_song_index)
    pygame.mixer.music.play()
    is_paused=0
    

def previous_song():
    global is_paused
    global current_song_index
    current_song_index = (current_song_index - 1) % len(playlist)
    load_song(current_song_index)
    pygame.mixer.music.play()
    is_paused=0

def volume_up():
    global volume
    volume = round(min(1.0, (volume + 0.1)),1)
    print(volume)
    pygame.mixer.music.set_volume(volume)
    # print(f"🔊 Volume: {int(volume * 100)}%")

def volume_down():
    global volume
    volume = round(max(0.0, (volume - 0.1)),1)
    pygame.mixer.music.set_volume(volume)
    print("volume",volume)
    # print(f"🔉 Volume: {int(volume * 100)}%")

# gesture recognition functions

# creazione di una zona di azione per la mano, meglio utilizzare 2 elementi: distanza tra polso e base medio e distanza tra polso e punta del pollice
# la zona di azione deve essere compresa tra i 50 e gli 80cm dalla telecamera

# LA ZONA DI AZIONE DOVREBBE ESSERE SOLO NELLA METÁ SUPERIORE DELLA RIPRESA?
min_dist_medio = 0.15 
min_dist_pollice = 0.20
min_dist_nocche=0.1

max_dist_medio = 0.35
max_dist_pollice = 0.45
max_dist_nocche=1


def get_min_dist_nocche(hand_landmarks):
    
    polso_medio= math.dist([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y ],[hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y ])
    polso_pollice= math.dist([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y ],[hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y ])
    return polso_medio*(180/11)+polso_pollice*(-240/11)+0.1


def zona_attiva(hand_landmarks):
    #se il gesto avviene in questa zona allora si esegue l'azione
    
    #distanze
    polso_medio= math.dist([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y ],[hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y ])
    polso_pollice= math.dist([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y ],[hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y ])
    dist_nocche=math.dist([hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y ],[hand_landmarks.landmark[17].x, hand_landmarks.landmark[17].y ])
    
    if ((polso_medio > min_dist_medio and\
        polso_medio <max_dist_medio) or\
        (polso_pollice > min_dist_pollice and\
        polso_pollice <max_dist_pollice)) and\
            (dist_nocche>0.08):
            
    # and\
    #     dist_nocche>get_min_dist_nocche(hand_landmarks):
        return True
    return False


def check_open_hand(hand_landmarks):
    all_y = [lm.y for lm in hand_landmarks.landmark]
    
    # IDs delle articolazioni delle dita (MediaPipe landmarks)
    finger_tips = [4, 8, 12, 16, 20]  # pollice, indice, medio, anulare, mignolo
    finger_pips = [2, 6, 10, 14, 18]   # articolazioni corrispondenti

    open_fingers = 0
    
    if hand_landmarks.landmark[0].y > min(all_y[1:]): #se il polso si trova più in basso rispetto a tutte le altre
        for i in range(5):
            tip = hand_landmarks.landmark[finger_tips[i]] #punta del dito corrente
            pip = hand_landmarks.landmark[finger_pips[i]] #articolazione del dito corrente
            
            # Per il pollice controlliamo la coordinata x
            if i == 0:  # pollice
                if (hand_landmarks.landmark[0].x < tip.x and tip.x > pip.x and hand_landmarks.landmark[1].x > hand_landmarks.landmark[0].x ) or \
                    (hand_landmarks.landmark[0].x > tip.x and tip.x < pip.x and hand_landmarks.landmark[1].x < hand_landmarks.landmark[0].x ):
                    open_fingers += 1
            # Per le altre dita controlliamo la coordinata y
            else:
                if tip.y < pip.y and\
                    abs(hand_landmarks.landmark[4].x-hand_landmarks.landmark[20].x)>0.1 and\
                    abs(hand_landmarks.landmark[0].y-hand_landmarks.landmark[12].y)>0.3    :  # la punta è sopra l'articolazione (dito aperto)
                    open_fingers += 1
        
        # Consideriamo la mano aperta se almeno 4 dita sono aperte (puoi cambiare a 5 per essere più preciso)
        return open_fingers >= 5
    else:
        return False


def check_thumbs_up(hand_landmarks):
    all_y = [lm.y for lm in hand_landmarks.landmark]
    all_x = [lm.x for lm in hand_landmarks.landmark]
    
    # new
    thumb_y = all_y[3:5] #joints 
    other_y = [all_y[0]]+all_y[5:] #all other joints 

    #con nocche laterali
    return max(thumb_y)<min(other_y) and not check_open_hand2(hand_landmarks)


def check_thumbs_down(hand_landmarks):
    all_y = [lm.y for lm in hand_landmarks.landmark]

    thumb_y = all_y[2:5] #joints 
    other_y = [all_y[0]]+all_y[5:] #all other joints 
    
    return max(other_y)<min(thumb_y) and not check_open_hand2(hand_landmarks)

def check_thumbs_rx(hand_landmarks):
    all_y = [lm.y for lm in hand_landmarks.landmark]
    all_x = [lm.x for lm in hand_landmarks.landmark]
    
    # new
    thumb_x = all_x[2:5]
    other_x = all_x[5:] #but not 0

    indici_nocche=[5, 9, 13, 17]
    indici_mid_nocche=[6, 10, 14, 18]

    all_but_nocche_thumb = [i for i in list(range(21)) if i not in indici_nocche+[2,3,4]]
    all_but_mid_nocche_thumb= [i for i in list(range(21)) if i not in indici_mid_nocche+[2,3,4]]
    
    print(all_but_nocche_thumb, all_but_mid_nocche_thumb)   
    indici_tips=[8, 12, 16, 20]
    mid_nocche=[all_y[i] for i in indici_mid_nocche]
    
    all_under_nocche = all(
        min(all_y[i] for i in indici_nocche) <= all_y[i]
        for i in all_but_nocche_thumb
    )
    
    all_under_mid_nocche = all(
        min(all_y[i] for i in indici_mid_nocche) <= all_y[i]
        for i in all_but_mid_nocche_thumb
    )

    if min(thumb_x) > max(other_x) and\
        hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x and\
        (all_under_nocche or all_under_mid_nocche):
        return True
    
    return False


def check_thumbs_rx_old(hand_landmarks):
    all_y = [lm.y for lm in hand_landmarks.landmark]
    all_x = [lm.x for lm in hand_landmarks.landmark]
    
    # new
    thumb_x = all_x[2:5]
    other_x = all_x[5:] #but not 0

    indici_nocche=[5, 9, 13, 17]
    nocche=[all_y[i] for i in indici_nocche]

    indici_tips=[8, 12, 16, 20]
    tips=[all_y[i] for i in indici_tips]

    dita_chiuse = any(
        all_y[tip] > all_y[nocca]
        for tip, nocca in zip(indici_tips, indici_nocche)
    )

    if min(thumb_x) > max(other_x) and\
        abs(hand_landmarks.landmark[4].y - hand_landmarks.landmark[17].y) < 0.2 and\
        abs(hand_landmarks.landmark[4].y - hand_landmarks.landmark[9].y) < 0.2 and\
            hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x and\
            dita_chiuse and\
            min(tips)>max(nocche):
        return True
    else:
        return False


def check_thumbs_sx(hand_landmarks):
    all_y = [lm.y for lm in hand_landmarks.landmark]
    all_x = [lm.x for lm in hand_landmarks.landmark]
    
    # new
    thumb_x = all_x[2:5]
    other_x = all_x[5:] #but not 0

    indici_nocche=[5, 9, 13, 17]
    indici_mid_nocche=[6, 10, 14, 18]

    all_but_nocche_thumb = [i for i in list(range(21)) if i not in indici_nocche+[2,4]]
    all_but_mid_nocche_thumb= [i for i in list(range(21)) if i not in indici_mid_nocche+[2,4]]
    
    indici_tips=[8, 12, 16, 20]
    mid_nocche=[all_y[i] for i in indici_mid_nocche]
    
    all_under_nocche = all(
        min(all_y[i] for i in indici_nocche) <= all_y[i]
        for i in all_but_nocche_thumb
    )
    
    all_under_mid_nocche = all(
        min(all_y[i] for i in indici_mid_nocche) <= all_y[i]
        for i in all_but_mid_nocche_thumb
    )

    if max(thumb_x) < min(other_x) and\
        hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x and\
        (all_under_nocche or all_under_mid_nocche):
        return True
    
    return False


def check_thumbs_sx_old(hand_landmarks):
    all_y = [lm.y for lm in hand_landmarks.landmark]
    all_x = [lm.x for lm in hand_landmarks.landmark]
    
    # new
    thumb_x = all_x[2:5]
    other_x = all_x[5:] #but not 0

    indici_nocche=[5, 9, 13, 17]
    indici_tips=[8, 12, 16, 20]
    
    nocche=[all_y[i] for i in indici_nocche]
    tips=[all_y[i] for i in indici_tips]
    
    dita_chiuse = any(
        all_y[tip] > all_y[nocca]
        for tip, nocca in zip(indici_tips, indici_nocche)
    )

    #inserire che il pollice deve stare a destra ripetto a tutti gli altri punti 4.x<all
    pollice_a_sx=all(
        hand_landmarks.landmark[4].x<=b
        for b in all_x
    )

    if  min(thumb_x) < max(other_x) and\
        abs(hand_landmarks.landmark[4].y - hand_landmarks.landmark[17].y) < 0.3 and\
        abs(hand_landmarks.landmark[4].y - hand_landmarks.landmark[9].y) < 0.5 and\
            hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x and\
            dita_chiuse and\
            pollice_a_sx and\
            min(tips)>max(nocche):
        return True
    else:
        return False


def check_open_hand2(hand_landmarks):
    #check se la mano è aperta per thumb
    # IDs delle articolazioni delle dita (MediaPipe landmarks)
    finger_tips = [4, 8, 12, 16, 20]  # pollice, indice, medio, anulare, mignolo
    finger_pips = [2, 6, 10, 14, 18]   # articolazioni corrispondenti

    open_fingers = 0
    
    for i in range(5):
        tip = hand_landmarks.landmark[finger_tips[i]] #punta del dito corrente
        pip = hand_landmarks.landmark[finger_pips[i]] #articolazione del dito corrente
        
        # Per il pollice controlliamo la coordinata x
        if i == 0:  # pollice
            if (hand_landmarks.landmark[0].x < tip.x and tip.x > pip.x and hand_landmarks.landmark[1].x > hand_landmarks.landmark[0].x ) or \
                (hand_landmarks.landmark[0].x > tip.x and tip.x < pip.x and hand_landmarks.landmark[1].x < hand_landmarks.landmark[0].x ):
                open_fingers += 0
        # Per le altre dita controlliamo la coordinata x
        else:
            if  abs(hand_landmarks.landmark[0].x-hand_landmarks.landmark[12].x)>0.2    :  # la punta è sopra l'articolazione (dito aperto)
                open_fingers += 1
    
    # Consideriamo la mano aperta se almeno 4 dita sono aperte (puoi cambiare a 5 per essere più preciso)
    return open_fingers >= 2

cooldown = 0.7 # Tempo di attesa tra i comandi (in secondi)
detection_delay = 0.4 # Delay prima di eseguire l'azione dopo il riconoscimento del gesto (in secondi)
cam=0

# Variabili globali
last_trigger_time = 0

detection_start_time = 0
is_detecting = False
current_gesture = None

# Nuove variabili per tener traccia delle ripetizioni consecutive
consecutive_thumbs_up_count = 0
consecutive_thumbs_down_count = 0
last_gesture = None


# Percorsi dei file audio per i feedback sonori
# Sostituisci questi percorsi con i file audio reali sul tuo sistema

path_start_sound = r"feedbacksounds/start_1.mp3"
path_start_1 = r"feedbacksounds/start_1.mp3"
path_start_i="feedbacksounds/vol_{i}.mp3"
path_completed_sound = r"feedbacksounds/stop.mp3"

# Initialize pygame mixer
pygame.mixer.init()
# Load the sound file
DETECTION_START_SOUND  = pygame.mixer.Sound(path_start_sound)
GESTURE_COMPLETED_SOUND = pygame.mixer.Sound(path_completed_sound)
# Set volume (0.0 to 1.0)
DETECTION_START_SOUND.set_volume(0)
GESTURE_COMPLETED_SOUND.set_volume(1)

def handle_gesture(name, action_fn, hand_landmarks, frame):
    global last_trigger_time, detection_start_time, is_detecting, current_gesture
    global consecutive_thumbs_up_count, consecutive_thumbs_down_count, last_gesture
    global is_paused
    
    # Se non stiamo già rilevando un gesto, inizia il processo di rilevamento
    if not is_detecting:
        is_detecting = True
        detection_start_time = time.time()
        current_gesture = name
        # Riproduci il suono di inizio rilevamento
        if name != "Vol DOWN" and name !="Vol UP":
            DETECTION_START_SOUND.play()
        # return
    
    # Se stiamo già rilevando e il gesto è cambiato, resetta il timer
    if current_gesture != name:
        is_detecting = True
        detection_start_time = time.time()
        current_gesture = name
        # Riproduci il suono di inizio rilevamento per il nuovo gesto
        if name != "Vol DOWN" and name !="Vol UP":
            DETECTION_START_SOUND.play()
        # return
    
    # Calcola il tempo trascorso dall'inizio del rilevamento
    elapsed_time = time.time() - detection_start_time
    
    # Posizione per visualizzare il nome del gesto e il cerchio
    x, y = int(hand_landmarks.landmark[0].x * frame.shape[1]), int(hand_landmarks.landmark[0].y * frame.shape[0])
    # Disegna il nome del gesto
    if not (name=="Play/Pause"):
        cv2.putText(frame, name.upper(), (400 , 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif is_paused:
        cv2.putText(frame, "PLAY", (400 , 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif not is_paused:
        cv2.putText(frame, "PAUSE", (400 , 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Disegna il cerchio di caricamento
    radius = 30
    center = (x, y-20)
    # Disegna il cerchio di sfondo (grigio)
    cv2.circle(frame, center, radius, (100, 100, 100), 3)
    # Calcola l'angolo in base al tempo trascorso (da 0 a 360 gradi)
    angle = min(elapsed_time / detection_delay * 360, 360)
    # Disegna l'arco di caricamento (verde)
    start_angle = -90  # Inizia dall'alto
    end_angle = start_angle + angle
    # Disegna l'arco di caricamento punto per punto
    for i in range(int(start_angle), int(end_angle)):
        rad = math.radians(i)
        x1 = int(center[0] + radius * math.cos(rad))
        y1 = int(center[1] + radius * math.sin(rad))
        cv2.circle(frame, (x1, y1), 1, (0, 255, 0), 3)
    
    # Se il tempo di attesa è completato, esegui l'azione
    if elapsed_time >= detection_delay:
        action_fn()
        # print(f"{name.upper()} RILEVATO!")
        # GESTURE_COMPLETED_SOUND.play()
    
        # Aggiorna i contatori per i gesti consecutivi
        if name == "Vol UP":
            if last_gesture == "Vol UP":
                consecutive_thumbs_up_count += 1
                if volume==1:
                    pygame.mixer.Sound(path_start_i.format(i=0)).play()
                else:
                    pygame.mixer.Sound(path_start_i.format(i = int(volume * 10))).play()                
            else:
                consecutive_thumbs_up_count = 1
                pygame.mixer.Sound(path_start_i.format(i = int(volume * 10))).play()
            consecutive_thumbs_down_count = 0
            # if consecutive_thumbs_up_count<=0 or consecutive_thumbs_down_count<=0:
            #     pygame.mixer.Sound(path_completed_sound).play()


        elif name == "Vol DOWN":
            if last_gesture == "Vol DOWN":
                consecutive_thumbs_down_count += 1
                if volume==0.0:
                    pygame.mixer.Sound(path_start_i.format(i=10)).play()
                else:
                    pygame.mixer.Sound(path_start_i.format(i = int(volume * 10))).play() 
            else:
                consecutive_thumbs_down_count = 1
                pygame.mixer.Sound(path_start_i.format(i = int(volume * 10))).play()
            consecutive_thumbs_up_count = 0
            # if consecutive_thumbs_up_count<=0 or consecutive_thumbs_down_count<=0:
            #     pygame.mixer.Sound(path_completed_sound).play()
        else:
            consecutive_thumbs_up_count = 0
            consecutive_thumbs_down_count = 0
            # if consecutive_thumbs_up_count<=0 or consecutive_thumbs_down_count<=0:
            pygame.mixer.Sound(path_completed_sound).play()
            
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


# Inizializzazione di MediaPipe e della webcam
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7)

cap = cv2.VideoCapture(cam)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # Resetta il flag di rilevamento se è passato troppo tempo senza rilevare lo stesso gesto
    if is_detecting and (time.time() - detection_start_time > detection_delay * 1.5):
        is_detecting = False
        current_gesture = None
    
    if results.multi_hand_landmarks and results.multi_handedness:
        
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            
            #check distanze
            dist_nocche= math.dist([hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y ],[hand_landmarks.landmark[17].x, hand_landmarks.landmark[17].y ])
            # polso_pollice=math.dist([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y ],[hand_landmarks.landmark[4].x, hand_landmarks.landmark[5].y ])
            
            # print("distanza")
            # print(dist_nocche)
            # print("distanza pollice")
            # print(polso_pollice)

            if zona_attiva(hand_landmarks):
                landmark_color = (0, 255, 0)  # Verde
                connection_color = (100, 100, 100)  
            if not zona_attiva(hand_landmarks):
                landmark_color = (0, 0, 255)  # Rosso
                connection_color = (0, 0, 255)  

            mp_draw.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=landmark_color, thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=connection_color, thickness=5, circle_radius=2))
            
            # Controlla se possiamo saltare il cooldown per i gesti "thumbs_up" e "thumbs_down"
            skip_cooldown = False
            
            if last_gesture == "Vol UP" and consecutive_thumbs_up_count >= 2:
                if check_thumbs_up(hand_landmarks):
                    skip_cooldown = True
            
            if last_gesture == "Vol DOWN" and consecutive_thumbs_down_count >= 2:
                if check_thumbs_down(hand_landmarks):
                    skip_cooldown = True

            # Verifica il cooldown o se possiamo saltarlo
            if zona_attiva(hand_landmarks) and ((time.time() - last_trigger_time) > cooldown or skip_cooldown):
                if check_open_hand(hand_landmarks) and estimate_hand_side(hand_landmarks.landmark, hand_handedness.classification[0].label) == "PALM" and ((estimate_hand_side(hand_landmarks.landmark, hand_handedness.classification[0].label) == "BACK")==False):
                    handle_gesture("Play/Pause", play_pause, hand_landmarks, frame)
                elif check_thumbs_up(hand_landmarks):
                    handle_gesture("Vol UP", volume_up, hand_landmarks, frame)
                elif check_thumbs_sx(hand_landmarks) :
                    handle_gesture("Prev", previous_song, hand_landmarks, frame)
                elif check_thumbs_rx(hand_landmarks) :
                    handle_gesture("Skip", next_song, hand_landmarks, frame)
                elif check_thumbs_down(hand_landmarks):
                    handle_gesture("Vol DOWN", volume_down, hand_landmarks, frame)
    
    # Visualizza il contatore di utilizzi consecutivi (opzionale)
    if consecutive_thumbs_up_count >= 2:
        cv2.putText(frame, f"Vol UP mode: {consecutive_thumbs_up_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if consecutive_thumbs_down_count >= 2:
        cv2.putText(frame, f"Vol DOWN mode: {consecutive_thumbs_down_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Handle music with gestures + palm ", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        pygame.mixer.music.pause()
        break

cap.release()
cv2.destroyAllWindows()