import cv2
import numpy as np
from collections import deque

# Load the binary image (replace 'binary_image.png' with your image file)
image = cv2.imread('D:/FIVETH SEM/system thinking/system_thinking_1/system_thinking/binary.png')

# Apply the Canny edge detection algorithm
edges = cv2.Canny(image, 30, 70)  # Adjust the thresholds as needed

# Display the edge map
cv2.imshow("Edge Map", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()