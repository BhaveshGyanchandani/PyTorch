# # .venv/Scripts/activate
# from gtts import gTTS
# import os

# # Text you want to convert to audio
# text = "Hello Bhavesh! This is your AI assistant speaking."

# # Choose language (English = 'en')
# language = 'en'

# # Create TTS object
# tts = gTTS(text=text, lang=language, slow=False)

# # Save as MP3
# tts.save("16_audio/output.mp3")

# # Play the audio (Windows)
# os.system("16_audio/output.mp3")

# For macOS use:
# os.system("afplay output.mp3")

# For Linux use:
# os.system("mpg321 output.mp3")


# First, activate your virtual environment in the terminal (not inside Python):
# .venv\Scripts\activate    <-- run this in terminal (NOT inside the code)

#2

# import pyttsx3

# # Initialize TTS engine
# engine = pyttsx3.init()

# # Set text
# text = "Hello Bhavesh! I can change my voice and speed too."

# # List available voices
# voices = engine.getProperty('voices')

    
# for i, voice in enumerate(voices):
#     print(f"{i}: {voice.name} - {voice.id}")

# # Choose a voice (try 0 or 1 based on output)
# engine.setProperty('voice', voices[1].id)

# # Change speed (default ~200)
# engine.setProperty('rate', 175)  # slower
# # Change volume (0.0 to 1.0)
# engine.setProperty('volume', 0.9)

# # Speak and save to file
# engine.say(text)
# engine.save_to_file(text, '16_audio/custom_voice.mp3')

# engine.runAndWait()


#3

import edge_tts
import asyncio

async def main():
    text = "Hello Bhavesh! I am using Microsoft's neural voice system."
    voice = "en-IN-PrabhatNeural"   # Try "en-IN-NeerjaNeural" for Indian female
    rate = "+5%"                 # Speed up
    output = "16_audio/edge_voice.mp3"

    tts = edge_tts.Communicate(text, voice=voice, rate=rate)
    await tts.save(output)
    print("Audio saved successfully!")

asyncio.run(main())


#4

# from openai import OpenAI
# client = OpenAI()

# speech = client.audio.speech.create(
#     model="gpt-4o-mini-tts",
#     voice="alloy",  # natural voice
#     input="Hello Bhavesh! This is GPT reading out loud."
# )

# with open("16_audio/openai_tts.mp3", "wb") as f:
#     f.write(speech.read())
