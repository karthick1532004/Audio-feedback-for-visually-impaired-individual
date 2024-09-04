import cv2
import torch
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Load the MiDaS model
model_path = "D:/FIVETH SEM/system thinking/PointNet-Plane-Detection-master/midas_v21_small_256.pt"
midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
midas_model.to("cuda")  # Move the model to GPU if available
midas_model.eval()

# Preprocess the input image
transform = Compose([Resize(256), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Read the input image
image = cv2.imread("D:/FIVETH SEM/system thinking/test5.jpg")  # Replace with the actual image path
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Preprocess the image for the model
input_image = transform(image).unsqueeze(0)
input_image = input_image.to("cuda")  # Move the input to GPU if available

# Run the MiDaS model to get the depth map
with torch.no_grad():
    prediction = midas_model(input_image)

# Post-process the depth map
depth_map = prediction.squeeze().cpu().numpy()

# Normalize the depth map to [0, 1] for visualization
depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))

# Display the depth map
cv2.imshow("Depth Map", depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
