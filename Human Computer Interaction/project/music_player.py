import pygame
import os

class MusicPlayer:
    def __init__(self, music_dir="songs"):
        pygame.init()
        pygame.mixer.init()
        self.music_dir = music_dir
        self.playlist = []
        self.current_song = 0
        self.volume = 0.5
        self.load_playlist()

    def load_playlist(self):
        self.playlist = [os.path.join(self.music_dir, f) for f in os.listdir(self.music_dir) if f.endswith(('.mp3', '.wav'))]
        self.playlist.sort()

    def play(self):
        pygame.mixer.music.load(self.playlist[self.current_song])
        pygame.mixer.music.play()
        pygame.mixer.music.set_volume(self.volume)

    def pause(self):
        pygame.mixer.music.pause()

    def resume(self):
        pygame.mixer.music.unpause()

    def next_track(self):
        self.current_song = (self.current_song + 1) % len(self.playlist)
        self.play()

    def previous_track(self):
        self.current_song = (self.current_song - 1) % len(self.playlist)
        self.play()

    def volume_up(self):
        self.volume = min(1.0, self.volume + 0.1)
        pygame.mixer.music.set_volume(self.volume)

    def volume_down(self):
        self.volume = max(0.0, self.volume - 0.1)
        pygame.mixer.music.set_volume(self.volume)