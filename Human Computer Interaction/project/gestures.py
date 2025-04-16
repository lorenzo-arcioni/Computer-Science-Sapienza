import cv2
import mediapipe as mp

class GestureController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect_gestures(self, frame):
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        gesture = None

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Get finger positions
            thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Calculate distances
            thumb_index_dist = abs(thumb_tip.x - index_tip.x) + abs(thumb_tip.y - index_tip.y)
            index_middle_dist = abs(index_tip.x - middle_tip.x) + abs(index_tip.y - middle_tip.y)

            # Gesture recognition logic
            if thumb_tip.y < index_tip.y and thumb_index_dist > 0.15:
                gesture = "volume_up"
            elif thumb_tip.y > index_tip.y and thumb_index_dist > 0.15:
                gesture = "volume_down"
            elif index_middle_dist < 0.05:
                gesture = "play_pause"
            elif index_tip.x < middle_tip.x:
                gesture = "next_track"
            else:
                gesture = "previous_track"

            # Draw landmarks
            self.mp_draw.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return frame, gesture