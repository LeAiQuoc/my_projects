import os
import subprocess

def convert_audio(input_file, output_file):
    try:
        # Convert the audio file using FFmpeg
        subprocess.run(['ffmpeg', '-i', input_file, output_file], check=True)
        print(f"Audio successfully converted: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting audio: {e}")

# Example: Convert output_vietnamese.wav to output_vietnamese.mp3
convert_audio("output_vietnamese.wav", "output_vietnamese.mp3")