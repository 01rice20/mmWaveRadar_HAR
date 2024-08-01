# Complement data 24
import os
import shutil
from PIL import Image
import random

def synchronize_images(spectrogram_folder, range_folder):
    cnt = 0
    range_folder2 = "./velocity/velocity_24_aug"
    for subfolder in os.listdir(spectrogram_folder):
        spectrogram_path = os.path.join(spectrogram_folder, subfolder)
        range_path = os.path.join(range_folder, subfolder)
        range_path2 = os.path.join(range_folder2, subfolder)
        # velocity_path = os.path.join(velocity_folder, subfolder)
        # print(spectrogram_path)

        if os.path.isdir(spectrogram_path) and os.path.isdir(range_path):
            spectrogram_images = set([os.path.splitext(f)[0] for f in os.listdir(spectrogram_path)])
            range_images = set([os.path.splitext(f.replace("_velocity", ""))[0] for f in os.listdir(range_path)])
            # velocity_images = set([os.path.splitext(f.replace("_velocity", ""))[0] for f in os.listdir(velocity_path)])

            missing_in_spectrogram = spectrogram_images - range_images
            missing_in_range = range_images - spectrogram_images

            os.makedirs(range_path2, exist_ok=True)
            cnt += len(missing_in_spectrogram)

            print(len(missing_in_spectrogram))
            
            for missing_image in missing_in_spectrogram:

                random_image = random.choice(os.listdir(range_path))
                source_path = os.path.join(range_path, random_image)
               
                target_name = f"{missing_image}_velocity{os.path.splitext(random_image)[1]}"
                target_path = os.path.join(range_path2, target_name)
               
                shutil.copy(source_path, target_path)

                print(f"Copied and renamed: {target_name}")
    print("cnt: ", cnt)
          

spectrogram_folder = "spectrogram/spectrogram_24"
range_folder = "./velocity/velocity_24"
synchronize_images(spectrogram_folder, range_folder)
