# voice_text.py

import speech_recognition as sr
import emotion_detection
import matplotlib.pyplot as plt
import librosa
import numpy as np


def convert_speech_to_text():
    # Initialize recognizer
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 0.8  # Stop listening after 2 seconds of silence

    # Capture audio from microphone
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source, phrase_time_limit=10)
        # Ensure audio_np has the correct number of time steps (e.g., 162) for the model

    emotion = emotion_detection.predict(audio)

    # Convert speech to text
    try:
        text = recognizer.recognize_google(audio)
        print("You said:", text)
        print("With emotion: ", emotion)
        return text
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
        return "Sorry, I could not understand the audio."
    except sr.RequestError:
        print("Could not request results from Google Speech Recognition service.")
        return "Could not request results from Google Speech Recognition service."

def convert_speech(audio_path):
    # Initialize recognizer
    recognizer = sr.Recognizer()

    try:
        # Load the audio file for recognition
        with sr.AudioFile(audio_path) as source:
            print("Processing audio file...")
            audio = recognizer.record(source)  # Read the entire audio file
            text = recognizer.recognize_google(audio)
            print("Recognized text:", text)
            return text
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
        return "Sorry, I could not understand the audio."
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service: {e}")
        return "Error in recognition service."

