import pygame
import os

class MusicPlayer:
    def __init__(self, music_folder="songs"):
        pygame.mixer.init()
        self.music_folder = music_folder
        self.songs = [f for f in os.listdir(music_folder) if f.endswith(".mp3")]
        self.index = 0
        self.load_song()

    def load_song(self):
        if self.songs:
            song_path = os.path.join(self.music_folder, self.songs[self.index])
            pygame.mixer.music.load(song_path)

    def play_pause(self):
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.pause()
        else:
            pygame.mixer.music.unpause() if pygame.mixer.music.get_pos() > 0 else pygame.mixer.music.play()

    def next(self):
        self.index = (self.index + 1) % len(self.songs)
        self.load_song()
        pygame.mixer.music.play()

    def prev(self):
        self.index = (self.index - 1) % len(self.songs)
        self.load_song()
        pygame.mixer.music.play()

    def volume_up(self):
        vol = min(pygame.mixer.music.get_volume() + 0.1, 1.0)
        pygame.mixer.music.set_volume(vol)

    def volume_down(self):
        vol = max(pygame.mixer.music.get_volume() - 0.1, 0.0)
        pygame.mixer.music.set_volume(vol)
