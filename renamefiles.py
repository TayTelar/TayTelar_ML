import os


def rename_images(directory,prefix):
    # List all files in the directory
    files = os.listdir(directory)

    # Filter out non-image files if necessary
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    # Sort files if you want a specific order
    image_files.sort()
    start =10
    # Rename files in sequence
    for index, file in enumerate(image_files):
        # Construct new file name
        new_name = f"{prefix}-{start + 1}{os.path.splitext(file)[1]}"
        start+=1
        # Full paths for old and new file names
        old_file = os.path.join(directory, file)
        new_file = os.path.join(directory, new_name)

        # Rename the file
        os.rename(old_file, new_file)
        print(f"Renamed {file} to {new_name}")


if __name__ == "__main__":
    # Replace 'your_directory_path' with the path to your directory
    directory_path = 'C:/Users/ADMIN/Downloads/new_pant_images/split_dataset/test'
    prefix="test"
    rename_images(directory_path,prefix)
