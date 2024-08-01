# Check if raw data and spectrogram got paired data
import os
import shutil

def count_matching_files():
    range_folder = "./velocity/velocity_24"
    image_folder = "./spectrogram/spectrogram_24"
    range_path2 = "./velocity/velocity_24_aug"
    sum = 0
    for subfolder in os.listdir(image_folder):
        current_image_folder = os.path.join(image_folder, subfolder)    # spectrogram/spectrogram_10/rawdata/10ghz/19_Scissors_gait
        current_raw_data_folder = os.path.join(range_folder, subfolder)  # range/range_24/19_Scissors_gait
        current_raw_data_folder2 = os.path.join(range_path2, subfolder)
        
        if os.path.isdir(current_image_folder) and os.path.isdir(current_raw_data_folder):
            # print("yes")
            image_files = [os.path.splitext(f)[0] for f in os.listdir(current_image_folder) if f.endswith('.png')]
            for root, dirs, files in os.walk(current_raw_data_folder):
                # print("file name: ", current_raw_data_folder)
                counter = 0
                for file in files:
                    raw_data_filename = os.path.splitext(file.replace("_velocity", ""))[0]
                    # print("raw_data_filename: ", raw_data_filename)
                    if raw_data_filename in image_files:
                        source_path = os.path.join(current_raw_data_folder, file)
                        target_path = os.path.join(current_raw_data_folder2, file)
                        shutil.copy(source_path, target_path)
                        print("Copied target_path: ", target_path)
                        counter += 1
                print("counter: ", counter)
                sum += counter
    print(f"Total matches found: {sum}")

count_matching_files()
