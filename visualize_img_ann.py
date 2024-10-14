import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_yolo_annotation(annotation_path):
    with open(annotation_path, 'r') as f:
        lines = f.readlines()

    annotations = []
    for line in lines:
        values = list(map(float, line.strip().split()))
        class_id, x_center, y_center, width, height = values[:5]
        keypoints = values[5:] if len(values) > 5 else []
        annotations.append((class_id, x_center, y_center, width, height, keypoints))

    return annotations


def visualize_annotations(image, annotations, keypoint_radius=6):
    h, w = image.shape[:2]

    for ann in annotations:
        class_id, x_center, y_center, width, height, keypoints = ann

        # Convert normalized coordinates to pixel coordinates
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Add class ID label
        label = f"Class: {int(class_id)}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Draw keypoints with a fixed size
        for i in range(0, len(keypoints), 3):
            kp_x = int(keypoints[i] * w)
            kp_y = int(keypoints[i + 1] * h)
            visibility = keypoints[i + 2]

            if visibility > 0:
                inner_color = (0, 255, 255) if visibility == 1 else (0, 0, 255)
                outer_color = (0, 0, 0)  # Black outline

                # Draw keypoints with a fixed radius
                cv2.circle(image, (kp_x, kp_y), keypoint_radius + 2, outer_color, -1)  # Outer circle
                cv2.circle(image, (kp_x, kp_y), keypoint_radius, inner_color, -1)  # Inner circle

    return image



def resize_image(image, max_size=800):
    h, w = image.shape[:2]
    if h > w:
        if h > max_size:
            w = int(w * max_size / h)
            h = max_size
    else:
        if w > max_size:
            h = int(h * max_size / w)
            w = max_size
    return cv2.resize(image, (w, h))


def process_folders(image_folder, annotation_folder):
    images = []
    titles = []

    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_folder, filename)
            annotation_filename = os.path.splitext(filename)[0] + '.txt'
            annotation_path = os.path.join(annotation_folder, annotation_filename)

            if os.path.exists(annotation_path):
                image = cv2.imread(image_path)
                annotations = read_yolo_annotation(annotation_path)

                # Visualize annotations on the original image
                visualized_image = visualize_annotations(image.copy(), annotations)

                # Resize the visualized image for display
                resized_image = resize_image(visualized_image)

                images.append(resized_image)
                titles.append(filename)

            else:
                print(f"Annotation file not found for {filename}")

    # Display images in a grid (4 images per window)
    for i in range(0, len(images), 4):
        plt.figure(figsize=(10, 10))
        for j in range(4):
            if i + j < len(images):
                plt.subplot(2, 2, j + 1)
                plt.imshow(cv2.cvtColor(images[i + j], cv2.COLOR_BGR2RGB))
                plt.title(titles[i + j])
                plt.axis('off')
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()

        # Wait for user input to show the next set of images
        input("Press Enter to visualize the next 4 images...")

        # Close the current figure
        plt.close()


# Usage
image_folder = 'pant_ds/images/train'  # Update with your image folder path
annotation_folder = 'pant_ds/labels/train'  # Update with your annotation folder path
process_folders(image_folder, annotation_folder)

