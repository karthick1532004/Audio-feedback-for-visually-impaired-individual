import cv2
from matplotlib import pyplot as plt
import sys
sys.path.append("c:/users/home/miniconda3/lib/site-packages/python-orb-slam3-folder")
from python_orb_slam3 import ORBExtractor



source = cv2.imread(r"D:\FIVETH SEM\system thinking\system_thinking_1\system_thinking\test10.jpg")
target = cv2.imread(r"D:\FIVETH SEM\system thinking\system_thinking_1\system_thinking\test10.jpg")

orb_extractor = ORBExtractor()

# Extract features from source image
source_keypoints, source_descriptors = orb_extractor.detectAndCompute(source)
target_keypoints, target_descriptors = orb_extractor.detectAndCompute(target)

# Match features
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(source_descriptors, target_descriptors)

# Draw matches
source_image = cv2.drawKeypoints(source, source_keypoints, None)
target_image = cv2.drawKeypoints(target, target_keypoints, None)
matches_image = cv2.drawMatches(source_image, source_keypoints, target_image, target_keypoints, matches, None)

# Show matches
plt.imshow(matches_image)
plt.show()