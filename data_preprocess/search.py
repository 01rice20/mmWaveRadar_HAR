# Check if raw data and spectrogram got paired data
import os
import shutil

def count_matching_files():
    raw_data_folder = "spectrogram/spectrogram_10"
    # raw_data_folder = "range/range_10"
    image_folder = "velocity/velocity_resize10"
    sum = 0
    for subfolder in os.listdir(image_folder):
        current_image_folder = os.path.join(image_folder, subfolder)    # spectrogram/spectrogram_10/rawdata/10ghz/19_Scissors_gait
        current_raw_data_folder = os.path.join(raw_data_folder, subfolder)  # rawdata/10ghz/19_Scissors_gait
        
        if os.path.isdir(current_image_folder) and os.path.isdir(current_raw_data_folder):
            # print("yes")
            image_files = [os.path.splitext(f.replace("_velocity_resized", ""))[0] for f in os.listdir(current_image_folder) if f.endswith('.png')]
            # print(image_files)
            for root, dirs, files in os.walk(current_raw_data_folder):
                # print("file name: ", current_raw_data_folder)
                counter = 0
                for file in files:
                    raw_data_filename = os.path.splitext(file)[0]
                    # raw_data_filename = os.path.splitext(file.replace("_range", ""))[0]
                    # print("raw_data_filename: ", raw_data_filename)
                    if raw_data_filename in image_files:
                        counter += 1
                print("counter: ", counter)
                sum += counter
    print(f"Total matches found: {sum}")

count_matching_files()