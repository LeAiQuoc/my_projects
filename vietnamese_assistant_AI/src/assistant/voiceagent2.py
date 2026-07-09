import pyaudio
import wave
import whisper
import ollama
import os
import time
import pygame
from pathlib import Path

# Initialize pygame and mixer
pygame.init()
pygame.mixer.init()  # Explicitly initialize the mixer
screen = pygame.display.set_mode((800, 480))  # Adjust to your display
font = pygame.font.Font(None, 36)

# Load Whisper (offline STT)
whisper_model = whisper.load_model("small")

# Ollama AI model
AI_MODEL = "mistral:7b"  # Install via `ollama pull mistral:7b`

class VietnameseVoiceAgent:
    def __init__(self):
        self.audio_file = "input.wav"
        self.output_audio = "output.wav"

    def record_audio(self, duration=5):
        """Record audio from mic."""
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        CHUNK = 1024

        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        print("Recording... Speak now.")
        frames = []

        for _ in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK, exception_on_overflow=False)
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
        if os.path.exists(self.audio_file):
            print(f"Audio file created: {self.audio_file}, Size: {os.path.getsize(self.audio_file)} bytes")
        else:
            print("Error: Audio file not created!")

    def speech_to_text(self):
        """Convert speech to text."""
        result = whisper_model.transcribe(self.audio_file, language="vi")
        return result["text"]

    def read_remarkable_text(self, file_path="remarkable_input.txt"):
        """Read text from reMarkable file."""
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        return None

    def generate_ai_response(self, input_text):
        """Generate response in Vietnamese."""
        prompt = f"Respond in Vietnamese to: {input_text}"
        response = ollama.generate(model=AI_MODEL, prompt=prompt)
        return response['response'].strip()

    def text_to_speech(self, text):
        """Convert text to speech with Piper on Windows."""
        with open("temp_text.txt", "w", encoding="utf-8") as f:
            f.write(text)
        cmd = f"C:\\piper\\piper.exe --model C:\\piper\\vi_VN-vietTTS.onnx --input temp_text.txt --output_file {self.output_audio}"
        os.system(cmd)
        os.remove("temp_text.txt")
        if os.path.exists(self.output_audio):
            print(f"Audio file created: {self.output_audio}, Size: {os.path.getsize(self.output_audio)} bytes")
            return self.output_audio
        else:
            print("Error: Audio file not created!")
            return None

    def play_audio(self):
        """Play audio non-blocking."""
        if os.path.exists(self.output_audio):
            try:
                pygame.mixer.music.load(self.output_audio)
                pygame.mixer.music.play()
                print("Playing audio...")
            except pygame.error as e:
                print(f"Error playing audio: {e}")
        else:
            print("Error: Audio file not found!")

    def display_text(self, text):
        """Show text on screen."""
        screen.fill((255, 255, 255))
        lines = text.split('\n')
        for i, line in enumerate(lines):
            surface = font.render(line, True, (0, 0, 0))
            screen.blit(surface, (10, 10 + i * 40))
        pygame.display.flip()

    def run(self):
        """Main loop."""
        clock = pygame.time.Clock()  # Control frame rate
        try:
            while True:
                clock.tick(30)  # 30 FPS for smoother response
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                input_text = None
                written_text = self.read_remarkable_text()
                if written_text:
                    input_text = written_text
                    print(f"Written input: {input_text}")
                else:
                    self.record_audio()
                    input_text = self.speech_to_text()
                    print(f"Transcribed: {input_text}")

                if input_text:
                    viet_response = self.generate_ai_response(input_text)
                    print(f"AI Response (VI): {viet_response}")
                    self.display_text(f"VI: {viet_response}")
                    audio_file = self.text_to_speech(viet_response)
                    if audio_file:
                        self.play_audio()
                        # Wait briefly to ensure audio starts, but don’t block
                        time.sleep(0.1)

                # Update display with prompt
                self.display_text("Nói hoặc viết tiếp...")

        except KeyboardInterrupt:
            print("\nExiting...")
            pygame.quit()

if __name__ == "__main__":
    agent = VietnameseVoiceAgent()
    agent.run()