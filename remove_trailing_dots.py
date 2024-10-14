import os


def remove_trailing_dots(label_folder):
    # Iterate through all files in the specified folder
    for filename in os.listdir(label_folder):
        # Check if the file is a .txt file
        if filename.endswith('.txt'):
            # Get the base name without the extension
            base_name = filename[:-4]  # Remove the '.txt' extension

            # Check if the base name ends with a dot
            if base_name.endswith('.'):
                # Create the new filename without the trailing dot
                new_base_name = base_name[:-1]  # Remove the trailing dot
                new_filename = new_base_name + '.txt'

                # Construct the full old and new file paths
                old_file_path = os.path.join(label_folder, filename)
                new_file_path = os.path.join(label_folder, new_filename)

                # Rename the file
                os.rename(old_file_path, new_file_path)
                print(f'Renamed: {filename} -> {new_filename}')


if __name__ == '__main__':
    # Specify the path to your label folder
    label_folder = 'm_ds/labels/val'
    remove_trailing_dots(label_folder)
