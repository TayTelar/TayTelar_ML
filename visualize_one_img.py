import cv2
import os
import matplotlib.pyplot as plt

def denormalize_coords(normalized_x, normalized_y, img_width, img_height):
    """
    Convert normalized coordinates to actual pixel values.
    """
    x = int(normalized_x * img_width)
    y = int(normalized_y * img_height)
    return x, y

def visualize_pose_on_image(image, annotation_data):
    img_height, img_width, _ = image.shape

    # Extract bounding box information
    class_id = int(annotation_data[0])
    bbox_cx, bbox_cy = float(annotation_data[1]), float(annotation_data[2])
    bbox_w, bbox_h = float(annotation_data[3]), float(annotation_data[4])

    # Convert normalized bbox to pixel values
    bbox_x_min = int((bbox_cx - bbox_w / 2) * img_width)
    bbox_y_min = int((bbox_cy - bbox_h / 2) * img_height)
    bbox_x_max = int((bbox_cx + bbox_w / 2) * img_width)
    bbox_y_max = int((bbox_cy + bbox_h / 2) * img_height)

    # Draw the bounding box
    cv2.rectangle(image, (bbox_x_min, bbox_y_min), (bbox_x_max, bbox_y_max), (255, 0, 0), 2)

    # Extract and draw keypoints
    keypoints = annotation_data[5:]
    for i in range(0, len(keypoints), 3):
        kp_x, kp_y = float(keypoints[i]), float(keypoints[i + 1])
        visibility = int(keypoints[i + 2])

        # Convert normalized keypoints to pixel values
        kp_x, kp_y = denormalize_coords(kp_x, kp_y, img_width, img_height)

        # Draw keypoints only if they are labeled (visibility = 0, 1, or 2)
        if visibility in [0, 1, 2]:
            color = (0, 255, 0) if visibility == 1 else (0, 0, 255)  # Green for visible, red for not visible
            cv2.circle(image, (kp_x, kp_y), 5, color, -1)

    return image, class_id

def visualize_single_image(image_path, annotation_path):
    # Read the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib

    # Read and parse the annotation
    with open(annotation_path, 'r') as f:
        annotation_data = f.readline().strip().split()

    # Annotate the image
    img_annotated, class_id = visualize_pose_on_image(img, annotation_data)

    # Display the image with annotations
    plt.imshow(img_annotated)
    plt.title(f"Class: {class_id}")
    plt.axis('off')
    plt.show()

def main():
    # Define the image and annotation paths
    image_path = 'pant_ds/images/train/train-6.jpg'  # Change to your image path
    annotation_path = 'bbox_cor/labels/train/train-6.txt'  # Change to your annotation path

    # Visualize the image with its annotation
    visualize_single_image(image_path, annotation_path)

if __name__ == "__main__":
    main()
