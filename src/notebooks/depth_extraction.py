import cv2
import numpy as np

def polygonal_floor_segmentation(image_path):
    # Load the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=5)

    # Create a mask to store the lines
    line_mask = np.zeros_like(gray)

    # Draw lines on the mask
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)

    # Find contours in the line mask
    contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area (the floor region)
    max_contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to get a polygonal boundary
    epsilon = 0.02 * cv2.arcLength(max_contour, True)
    polygonal_boundary = cv2.approxPolyDP(max_contour, epsilon, True)

    # Draw the polygonal boundary on the original image
    cv2.polylines(img, [polygonal_boundary], isClosed=True, color=(0, 0, 255), thickness=2)

    return img

if __name__ == "__main__":
    image_path = "D:/FIVETH SEM/system thinking/test44.jpg"
    segmented_img = polygonal_floor_segmentation(image_path)

    # Display the original and segmented images
    cv2.imshow("Original Image", cv2.imread(image_path))
    cv2.imshow("Segmented Image", segmented_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# read two input images as grayscale images
#imgL = cv2.imread('D:/FIVETH SEM/system thinking/depthtest1l.jpg',0)
#imgR = cv2.imread('D:/FIVETH SEM/system thinking/depthtest1r2.jpg',0)

