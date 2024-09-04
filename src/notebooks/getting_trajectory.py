import numpy as np
import cv2
import math
from collections import deque
import pyttsx3

prevDeltas = deque(maxlen=6)

def getSteeringCommand(windowCenter, centerPoints, img):
    shortestDelta = 999
    side = "Center"  # Initialize side as Center

    above_point = None
    below_point = None

    for centerPoint in centerPoints:
        delta = round(calculateDistance(windowCenter[0], windowCenter[1], centerPoint[0], centerPoint[1]), 2)

        # Determine whether the point is on the left or right side
        if centerPoint[0] < windowCenter[0]:
            side = "Left"
        else:
            side = "Right"

        # Update above and below points
        if centerPoint[1] < windowCenter[1]:
            if above_point is None or centerPoint[1] > above_point[1]:
                above_point = centerPoint
        elif centerPoint[1] > windowCenter[1]:
            if below_point is None or centerPoint[1] < below_point[1]:
                below_point = centerPoint

    a = above_point
    b = below_point

    if a is not None and b is not None:
        # Calculate the line equation (y = mx + c) for the points a and b
        m = (b[1] - a[1]) / (b[0] - a[0])
        c = a[1] - m * a[0]

        # Calculate the x-coordinate of the point on the line with the same y-axis as windowCenter
        x_on_line = (windowCenter[1] - c) / m

        # Calculate the distance between windowCenter and the point on the line
        distance_to_line = calculateDistance(windowCenter[0], windowCenter[1], x_on_line, windowCenter[1])

        prevDeltas.append(distance_to_line)
    
        avg_distance_to_line = round(sum(prevDeltas) / len(prevDeltas))

        # Print and display the distance to the line and side
        print("Distance to Line:", avg_distance_to_line)
        cv2.putText(img, f"Distance to Line: {avg_distance_to_line}, Side: {side}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)

        cv2.circle(img, windowCenter, 5, (0, 255, 255), -1)

        # Speak the distance and side
        speak_distance(avg_distance_to_line, side)
    else:
        print("No valid above and below points found.")

    return img
def calculateDistance(x1, y1, x2, y2):  
     dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  
     return dist

def speak_distance(distance, side):
    """
    This function takes the distance and side as input and 
    compares if the distance is less than 20. 
    If the distance is less than 20, the function will 
    tell the speaker to move straight. 
    If the distance is greater than 20, the function will 
    tell the speaker to turn right.
    """

    # Create a pyttsx3 object
    speaker = pyttsx3.init()

    # Set the volume of the speaker
    speaker.setProperty('volume', 1.0)

    # Set the rate of the speaker
    speaker.setProperty('rate', 150)

    try:
        distance = float(distance)  # Convert input to float
        # If the distance is less than 20, tell the speaker to move straight
        if distance < 80:
            speaker.say("Move straight")
        elif distance > 80:
            if side == "Right":
               speaker.say("move right")
            elif side == "Left":
               speaker.say("move left")
    except ValueError:
        # Handle invalid input for distance
        speaker.say("Invalid input for distance")

    try:
        # Handle invalid input for side
        if side.lower() not in ["left", "right"]:
            speaker.say("Invalid input for side")
    except AttributeError:
        pass

    # Run the text-to-speech engine
    speaker.runAndWait()