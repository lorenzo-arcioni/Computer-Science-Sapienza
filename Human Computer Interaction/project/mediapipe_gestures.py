import cv2
import mediapipe as mp
import time

class HandGestureDetector:
    def __init__(self, max_hands=1, detection_conf=0.7):
        self.hands = mp.solutions.hands.Hands(max_num_hands=max_hands, min_detection_confidence=detection_conf)
        self.mp_draw = mp.solutions.drawing_utils
        self.last_gesture = None
        self.last_time = time.time()

    def detect(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        gesture = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark

                # Semplici gesture
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                middle_tip = landmarks[12]

                # Posizioni base
                if index_tip.y < middle_tip.y and index_tip.y < thumb_tip.y:
                    gesture = "volume_up"
                elif thumb_tip.y > index_tip.y and thumb_tip.y > middle_tip.y:
                    gesture = "volume_down"
                elif abs(index_tip.x - thumb_tip.x) > 0.3:
                    if index_tip.x > thumb_tip.x:
                        gesture = "next"
                    else:
                        gesture = "prev"
                else:
                    gesture = "play_pause"

        now = time.time()
        if gesture and gesture != self.last_gesture and now - self.last_time > 1:
            self.last_gesture = gesture
            self.last_time = now
            return gesture
        return None
