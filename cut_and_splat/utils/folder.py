import os
import shutil


supported_images = ['jpg', 'jpeg', 'png', 'gif', 'bmp']


def create_or_empty_folders(output_path: str):
    # Creating or emptying folders
    render_path = os.path.join(output_path, "renders")
    mask_path = os.path.join(output_path, "masks")

    shutil.rmtree(render_path, ignore_errors=True)
    shutil.rmtree(mask_path, ignore_errors=True)
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)

    return render_path, mask_path


def find_images_in_folder(folder_path: str) -> list:
    """
    Return the path to all images in the given folder and its sub-folders
    """
    image_paths = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any([file.lower().endswith(extension) for extension in supported_images]):
                image_paths.append(os.path.join(root, file))

    return image_paths