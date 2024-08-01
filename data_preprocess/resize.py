# Resize image to 128*128
import os
import shutil
from PIL import Image

def count_matching_files():
    image_folder = "velocity/velocity_10"
    resize_folder = "velocity/velocity_resize10"

    subfolders = [f.path for f in os.scandir(image_folder) if f.is_dir()]
    for subfolder in subfolders:
        # print("subfolder", subfolder)
        folder_name = os.path.basename(subfolder)
        # print("folder_name", folder_name)
        target_subfolder = os.path.join(resize_folder, folder_name)
        os.makedirs(target_subfolder, exist_ok=True)
        files = [f.path for f in os.scandir(subfolder) if f.is_file()]
        
        for file in files:
            # shutil.copy(file, target_subfolder)
            with Image.open(file) as img:
                resized_img = img.resize((128, 128))
                file_name, file_extension = os.path.splitext(os.path.basename(file))
                target_path = os.path.join(target_subfolder, f"{file_name}_resized{file_extension}")
                # print("target_path: ", target_path)
                resized_img.save(target_path)

count_matching_files()