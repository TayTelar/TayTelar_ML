import ultralytics
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np

model =YOLO("trained_phone_keypoint.pt")
# Path to the image for inference
image_path = "thai-nguyen-HKwbLKMLQ1g-unsplash.jpg"

# Run inference
res = model(image_path)

# Get the first result
result = res[0]

# Load the image
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Unable to load image from {image_path}")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Create a figure and axis
fig, ax = plt.subplots(1, figsize=(12, 9))

# Display the image
ax.imshow(img)

# Plot bounding box
boxes = result.boxes
if len(boxes) > 0:
    box = boxes[0]  # Get the first box
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)

# Plot keypoints
keypoints = result.keypoints
if keypoints is not None and len(keypoints) > 0:
    kpts = keypoints[0].xy[0].tolist()  # Get k eypoints of the first detection
    for kpt in kpts:
        print(f"( {kpt[0]} , {kpt[1]})")
        ax.plot(kpt[0], kpt[1], 'go', markersize=10)  # 'go' means green circle

# Remove axis ticks
ax.set_xticks([])
ax.set_yticks([])

# Show the plot
plt.tight_layout()
plt.show()
if len(boxes) > 0:
    print("Bounding box coordinates:", boxes[0].xyxy[0].tolist())
if keypoints is not None and len(keypoints) > 0:
    print("Keypoint coordinates:", keypoints[0].xy[0].tolist())