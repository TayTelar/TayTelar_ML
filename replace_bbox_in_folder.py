import os

# Paths to your folders
folder_with_bboxes = 'bbox_cor/train'  # Folder with bbox annotations
folder_with_keypoints = 'pant_ds/labels/train'  # Folder with keypoint annotations

# Function to read YOLO formatted annotations (only bounding boxes)
def read_yolo_bboxes(file_path):
    bboxes = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_index = parts[0]
            bbox = list(map(float, parts[1:5]))  # [x_center, y_center, width, height]
            bboxes[class_index] = bbox
    return bboxes

# Function to update the bounding boxes in the keypoint annotation file
def update_bboxes_in_keypoints(file_path, new_bboxes):
    updated_lines = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_index = parts[0]

            if class_index in new_bboxes:
                # Replace only the bbox part of the line
                updated_line = f"{class_index} {' '.join(map(str, new_bboxes[class_index]))} {' '.join(parts[5:])}\n"
                updated_lines.append(updated_line)
            else:
                # Keep the line unchanged (for keypoints)
                updated_lines.append(line)
    return updated_lines

# Main process
def main():
    # Iterate over all annotation files in the second folder
    for filename in os.listdir(folder_with_keypoints):
        if filename.endswith('.txt'):
            # Construct paths
            bbox_file_path = os.path.join(folder_with_bboxes, filename)
            keypoint_file_path = os.path.join(folder_with_keypoints, filename)

            if os.path.exists(bbox_file_path):
                # Read the new bounding boxes
                new_bboxes = read_yolo_bboxes(bbox_file_path)

                # Update the keypoint file with the new bounding boxes
                updated_lines = update_bboxes_in_keypoints(keypoint_file_path, new_bboxes)

                # Write the updated lines back to the file
                with open(keypoint_file_path, 'w') as file:
                    file.writelines(updated_lines)

if __name__ == "__main__":
    main()