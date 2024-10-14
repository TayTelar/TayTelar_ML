import os
import ultralytics
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Load the model
model = YOLO("best.pt")

# Path to the folder containing images
image_folder = "C:/Users/ADMIN/Downloads/new_pant_images/split_dataset/test"
#
#pant_ds/images/test/

# Loop through all images in the folder
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)

    # Run inference
    res = model(image_path)

    # Get the first result
    result = res[0]

    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Unable to load image from {image_path}")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create a figure and axis
    fig, ax = plt.subplots(1, figsize=(12, 9))

    # Display the image
    ax.imshow(img)

    # Plot bounding boxes
    boxes = result.boxes
    if len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

    # Plot keypoints
    keypoints = result.keypoints
    if keypoints is not None and len(keypoints) > 0:
        for kpts in keypoints:
            kpts = kpts.xy[0].tolist()  # Get keypoints of the first detection
            for kpt in kpts:
                print(f"( {kpt[0]} , {kpt[1]})")
                ax.plot(kpt[0], kpt[1], 'go', markersize=5)  # 'go' means green circle

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Show the plot
    plt.tight_layout()
    plt.title(f"Results for {image_name}")
    plt.show()

    # Print bounding box and keypoint coordinates
    if len(boxes) > 0:
        print("Bounding box coordinates:", boxes[0].xyxy[0].tolist())
    if keypoints is not None and len(keypoints) > 0:
        print("Keypoint coordinates:", keypoints[0].xy[0].tolist())
