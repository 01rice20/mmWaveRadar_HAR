# Concate all frequency images
import os
import shutil

source_folders = ['velocity/velocity_resize10', 'velocity/velocity_24_aug', 'velocity/velocity_resize77']
target_folder = 'velocity/velocity_resize111'
os.makedirs(target_folder, exist_ok=True)

for source_folder in source_folders:
    subfolders = [f.path for f in os.scandir(source_folder) if f.is_dir()]
    # print("subfolders", subfolders)
    for subfolder in subfolders:
        # print("subfolder", subfolder)
        folder_name = os.path.basename(subfolder)
        # print("folder_name", folder_name)
        target_subfolder = os.path.join(target_folder, folder_name)
        os.makedirs(target_subfolder, exist_ok=True)
        files = [f.path for f in os.scandir(subfolder) if f.is_file()]
        
        for file in files:
            shutil.copy(file, target_subfolder)

print("Finished")
