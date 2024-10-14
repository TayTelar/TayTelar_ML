import os


def check_image_label_pairs(image_folder, label_folder):
    images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    label_file_names = {os.path.splitext(f)[0] for f in os.listdir(label_folder) if f.lower().endswith('.txt')}

    for image in images:
        image_name = os.path.splitext(image)[0]
        if image_name not in label_file_names:
            print(f'No corresponding label file for image: {image}')



image_folder = 'pant_ds/images/train'
label_folder = 'pant_ds/labels/train'
check_image_label_pairs(image_folder, label_folder)
