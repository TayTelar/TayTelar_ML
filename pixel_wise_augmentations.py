import os
import cv2
import albumentations as A
import random

# Define a list of pixel-wise augmentations
augmentations = [
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    A.RandomGamma(p=0.5),
    A.CLAHE(p=0.5),
]


def augment_image(image):
    # Randomly choose an augmentation from the list
    augmentation = random.choice(augmentations)
    augmented = augmentation(image=image)
    return augmented['image']


def augment_image_and_annotations(image_path, annotation_path, output_image_dir, output_label_dir, new_index):
    # Load the image
    image = cv2.imread(image_path)

    # Perform the augmentation
    augmented_image = augment_image(image)

    # Save the augmented image
    output_image_path = os.path.join(output_image_dir, f'aug_train_{new_index}.jpg')
    cv2.imwrite(output_image_path, augmented_image)

    # Copy the annotation file with the new name
    output_annotation_path = os.path.join(output_label_dir, f'aug_train_{new_index}.txt')
    with open(output_annotation_path, 'w') as f:
        with open(annotation_path, 'r') as orig_f:
            f.write(orig_f.read())


def main():
    image_folder = 'C:/Users/ADMIN/Downloads/d/temp_ds/image'  # Folder containing images
    annotation_folder = 'C:/Users/ADMIN/Downloads/d/temp_ds/labels'  # Folder containing YOLO annotations
    output_image_dir = 'pant_ds/images/train'  # Output folder for augmented images
    output_label_dir = 'pant_ds/labels/train'  # Output folder for augmented annotations

    # Create output directories if they do not exist
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    new_index = 149 # Initialize image index for naming

    # Process each image and its corresponding annotation
    for image_name in os.listdir(image_folder):
        if image_name.endswith('.jpg'):
            image_path = os.path.join(image_folder, image_name)
            annotation_path = os.path.join(annotation_folder, image_name.replace('.jpg', '.txt'))

            # Augment and save the image and annotation
            augment_image_and_annotations(image_path, annotation_path, output_image_dir, output_label_dir, new_index)
            new_index += 1  # Increment index for the next augmented image


if __name__ == "__main__":
    main()
