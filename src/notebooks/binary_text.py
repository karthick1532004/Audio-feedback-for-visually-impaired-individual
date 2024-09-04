import cv2
import numpy as np

# Load the binary image
binary_image_path = 'D:/FIVETH SEM/system thinking/vision_based_robot_navigation-master/final_image_with_centerline1.png'
binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)

# Convert binary image to a NumPy array
binary_array = np.array(binary_image)

# Save the NumPy array to a text file
output_file_path = 'binary_image1.txt'
np.savetxt(output_file_path, binary_array, delimiter=',', fmt='%d')

print(f"Binary image saved to {output_file_path}")
