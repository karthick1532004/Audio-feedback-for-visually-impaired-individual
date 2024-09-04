from gtts import gTTS
import pygame
import numpy as np  # Import numpy for distance calculation

def distance_and_direction(mytext, distance, side):

    # Check if the distance is greater than 70
    if distance > 70:
        if side == "Left":
            mytext += " Move right."
        elif side == "Right":
            mytext += " Move left."

    language = 'en'

    myobj = gTTS(text=mytext, lang=language, slow=False)
    myobj.save("welcome.mp3")

    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load("welcome.mp3")
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        continue

# Example usage:
distance_value = 80  # Replace this with your actual distance value
side_value = "Left"  # Replace this with your actual side value (can be "Left" or "Right")
text_to_speak = "The distance is {} units on the {} side.".format(distance_value, side_value)

distance_and_direction(text_to_speak, distance_value, side_value)
