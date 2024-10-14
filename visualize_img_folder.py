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

def plot_images_in_grid(images, annotations, image_paths, grid_size=8):
    """
    Visualizes the images and their annotations in a grid layout.
    """
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 16))
    axes = axes.flatten()

    for i, (img_path, annotation_path) in enumerate(zip(image_paths, annotations)):
        if i >= grid_size * grid_size:
            break  # Limit to grid_size x grid_size images

        # Read the image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib

        # Read and parse the annotation
        with open(annotation_path, 'r') as f:
            annotation_data = f.readline().split()

        # Annotate the image
        img_annotated, class_id = visualize_pose_on_image(img, annotation_data)

        # Plot the image with annotations
        axes[i].imshow(img_annotated)
        axes[i].set_title(f"Class: {class_id}", fontsize=12)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def main():
    # Define the folders containing images and annotations
    image_folder = 'pant_ds/images/train'  # Folder containing images
    annotation_folder = 'bbox_cor/labels/train'  # Folder containing annotation txt files

    # List of image files and their corresponding annotation files
    image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')])
    annotation_files = sorted([os.path.join(annotation_folder, f.replace('.jpg', '.txt')) for f in os.listdir(image_folder) if f.endswith('.jpg')])

    # Visualize images in a 2x2 grid
    plot_images_in_grid(image_files, annotation_files, image_files, grid_size=2)

if __name__ == "__main__":
    main()
