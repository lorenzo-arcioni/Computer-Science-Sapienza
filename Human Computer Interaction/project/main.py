from kivy.app import App
from kivy.uix.label import Label
from kivy.clock import Clock
import threading
import cv2
from music_player import MusicPlayer
from mediapipe_gestures import HandGestureDetector

class GestureApp(App):
    def build(self):
        self.label = Label(text="Controllo musicale con gesture")
        Clock.schedule_once(lambda dt: threading.Thread(target=self.cv_loop).start(), 1)
        return self.label

    def cv_loop(self):
        player = MusicPlayer()
        detector = HandGestureDetector()
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gesture = detector.detect(frame)

            if gesture == "next":
                player.next()
            elif gesture == "prev":
                player.prev()
            elif gesture == "play_pause":
                player.play_pause()
            elif gesture == "volume_up":
                player.volume_up()
            elif gesture == "volume_down":
                player.volume_down()

        cap.release()

if __name__ == "__main__":
    GestureApp().run()