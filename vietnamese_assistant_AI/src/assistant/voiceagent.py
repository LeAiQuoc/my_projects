import numpy as np
import pyaudio
import wave
import whisper
import ollama
from gtts import gTTS
from argostranslate import package, translate
from argostranslate.translate import get_installed_languages
import pygame
import os
import time
from pathlib import Path

# Initialize pygame for audio playback and display
pygame.init()
screen = pygame.display.set_mode((800, 480))  # Adjust to your display size
font = pygame.font.Font(None, 36)

# Load Whisper model (offline STT)
whisper_model = whisper.load_model("small")  # "small" for Pi, "medium" if RAM allows


installed_languages = get_installed_languages()
for lang in installed_languages:
    print(f"Code: {lang.code}, Name: {lang.name}")

# Load Argos Translate for Vietnamese-English
package.update_package_index()
available_packages = package.get_available_packages()
vi_en_package = next((pkg for pkg in available_packages if pkg.from_code == "vi" and pkg.to_code == "en"), None)
if vi_en_package:
    package.install_from_path(vi_en_package.download())
installed_languages = translate.get_installed_languages()
vi_lang = next(lang for lang in installed_languages if lang.code == "vi")
en_lang = next(lang for lang in installed_languages if lang.code == "en")
translator = vi_lang.get_translation(en_lang)



# Ollama model (install locally, e.g., 'mistral:7b')
AI_MODEL = "mistral:7b"

class VietnameseVoiceAgent:
    def __init__(self):
        self.recording_duration = 5  # Seconds to record
        self.audio_file = "input.wav"
        self.output_audio = "output.wav"

    def record_audio(self):
        """Record audio from microphone."""
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        CHUNK = 1024

        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        print("Recording... Speak now.")
        frames = []

        for _ in range(0, int(RATE / CHUNK * self.recording_duration)):
            data = stream.read(CHUNK)
            frames.append(data)

        print("Finished recording.")
        stream.stop_stream()
        stream.close()
        p.terminate()

        with wave.open(self.audio_file, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))

    def speech_to_text(self):
        """Convert Vietnamese speech to text using Whisper."""
        result = whisper_model.transcribe(self.audio_file, language="vi")
        return result["text"]

    def read_remarkable_text(self, file_path="remarkable_input.txt"):
        """Read digitized text from reMarkable (assumes file sync)."""
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        return None

    def generate_ai_response(self, input_text):
        """Generate Vietnamese response using Ollama."""
        prompt = f"Respond in Vietnamese to: {input_text}"
        response = ollama.generate(model=AI_MODEL, prompt=prompt)
        return response['response']

    def translate_viet_to_eng(self, text):
        """Translate Vietnamese to English."""
        return translate(text, from_code="vi", to_code="en")

    def text_to_speech(self, text):
        """Convert Vietnamese text to speech using Piper (or eSpeak NG)."""
        # Using Piper TTS (install via: https://github.com/rhasspy/piper)
        os.system(f"echo '{text}' | piper --model vi_VN --output_file {self.output_audio}")
        # Alternative with eSpeak NG: os.system(f"espeak -v vi '{text}' -w {self.output_audio}")
        return self.output_audio
    
    # Alternative text_to_speech using gTTS
    def convert_text_to_speech(self, text):
        """
        Convert the given text to speech using gTTS (Vietnamese).
        """
        try:
            tts = gTTS(text, lang="vi")  # Set language to Vietnamese
            audio_file_path = "output_vietnamese.mp3"
            tts.save(audio_file_path)
            print(f"Audio saved as {audio_file_path}")
            return audio_file_path
        except Exception as e:
            print(f"Error generating speech: {e}")
            return None


    def display_text(self, text):
        """Display text on screen."""
        screen.fill((255, 255, 255))  # White background
        text_surface = font.render(text, True, (0, 0, 0))  # Black text
        screen.blit(text_surface, (10, 10))
        pygame.display.flip()

    def play_audio(self):
        """Play audio file."""
        pygame.mixer.music.load(self.output_audio)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)


    def run(self):
        """Main loop."""
        try:
            while True:
                # Check for reMarkable text input
                written_text = self.read_remarkable_text()
                if written_text:
                    print(f"Written input: {written_text}")
                    input_text = written_text
                else:
                    # Record and transcribe speech
                    self.record_audio()
                    input_text = self.speech_to_text()
                    print(f"Transcribed: {input_text}")

                if input_text:
                    # Generate AI response in Vietnamese
                    viet_response = self.generate_ai_response(input_text)
                    print(f"AI Response: {viet_response}")

                    # Translate to English (optional)
                    eng_response = self.translate_viet_to_eng(viet_response)
                    print(f"English Translation: {eng_response}")

                    # Display both
                    self.display_text(f"VI: {viet_response}\nEN: {eng_response}")

                    # Convert to speech and play
                    audio_file = self.convert_text_to_speech(viet_response)
                    self.play_audio()

                time.sleep(1)  # Avoid overwhelming the Pi

        except KeyboardInterrupt:
            print("\nExiting...")
            pygame.quit()

if __name__ == "__main__":
    agent = VietnameseVoiceAgent()
    agent.run()