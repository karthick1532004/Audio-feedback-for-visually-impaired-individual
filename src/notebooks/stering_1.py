import numpy as np
import cv2
import math
from collections import deque

prevDeltas = deque(maxlen=6)

def getSteeringCommand(windowCenter, centerPoints, img):
    shortestDelta = 999
    nearestPoint = None

    # Initialize variables outside the if statement
    angle_to_reference_line_deg = 0
    avgShortestDelta = 0

    for centerPoint in centerPoints:
        if centerPoint[1] < windowCenter[1]:
            delta = round(calculateDistance(windowCenter[0], windowCenter[1], centerPoint[0], centerPoint[1]), 2)
            if delta < shortestDelta:
                shortestDelta = delta
                nearestPoint = centerPoint

    if nearestPoint is not None:
        # Calculate the angle between the horizontal reference line
        # where right side is 0 degrees and left side is 180 degrees
        horizontal_reference_line = [(0, windowCenter[1]), (img.shape[1], windowCenter[1])]

        # Calculate the angle in radians
        angle_rad = math.atan2(nearestPoint[1] - windowCenter[1], nearestPoint[0] - windowCenter[0])
        angle_deg = math.degrees(angle_rad)

        # Calculate the angle to the reference line (0 to 180 degrees)
        angle_to_reference_line_rad = math.atan2(nearestPoint[1] - windowCenter[1], nearestPoint[0] - windowCenter[0]) - math.atan2(windowCenter[1] - windowCenter[1], windowCenter[0] - windowCenter[0])
        angle_to_reference_line_deg = math.degrees(angle_to_reference_line_rad)

        # Convert the angle to the range of 0 to 180 degrees
        if angle_to_reference_line_deg < 0:
            angle_to_reference_line_deg += 180

        a = (centerPoints[-1][0], centerPoints[-1][1])
        b = (centerPoints[0][0], centerPoints[0][1])

        shortestDelta *= np.sign((b[0] - a[0]) * (windowCenter[1] - a[1]) - (b[1] - a[1]) * (windowCenter[0] - a[0]))

        prevDeltas.append(shortestDelta)
        avgShortestDelta = round(sum(prevDeltas) / len(prevDeltas))

        cv2.putText(img, "Delta: " + str(avgShortestDelta), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)

        # Draw the horizontal reference line
        cv2.line(img, horizontal_reference_line[0], horizontal_reference_line[1], (0, 255, 0), 2)

        # Draw a line from windowCenter to the nearest point above the windowCenter
        cv2.line(img, windowCenter, nearestPoint, (0, 255, 0), 2)

        if (0 <= angle_to_reference_line_deg <= 20) or (160 <= angle_to_reference_line_deg <= 180):
            # Calculate the next shortest distance point
            next_shortest_delta = 999
            next_nearest_point = None
            for centerPoint in centerPoints:
                if centerPoint[1] < windowCenter[1] and centerPoint != nearestPoint:
                    delta = round(calculateDistance(windowCenter[0], windowCenter[1], centerPoint[0], centerPoint[1]), 2)
                    if delta < next_shortest_delta:
                        next_shortest_delta = delta
                        next_nearest_point = centerPoint
            if next_nearest_point is not None:
                angle_to_reference_line_rad = math.atan2(next_nearest_point[1] - windowCenter[1], next_nearest_point[0] - windowCenter[0]) - math.atan2(windowCenter[1] - windowCenter[1], windowCenter[0] - windowCenter[0])
                angle_to_reference_line_deg = math.degrees(angle_to_reference_line_rad)
                if angle_to_reference_line_deg < 0:
                    angle_to_reference_line_deg += 180

        # You can add more conditions for other specific angle ranges if needed.

    # Move these lines outside the if statement
    else:
        angle_deg = 0
        avgShortestDelta = 0

    cv2.circle(img, windowCenter, 5, (0, 255, 255), -1)

    return img, angle_to_reference_line_deg, avgShortestDelta

def calculateDistance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist
