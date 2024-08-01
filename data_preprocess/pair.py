# Check if velocity and range got paired images from spectrogram, print the missing image if not
import os

def find_missing_images():
    spectrogram_folder = "./spectrogram/spectrogram_10"
    # range_folder = "./range/range_10"
    velocity_folder = "./velocity/velocity_10"

    for subfolder in os.listdir(spectrogram_folder):
        spectrogram_path = os.path.join(spectrogram_folder, subfolder)
        # range_path = os.path.join(range_folder, subfolder)
        velocity_path = os.path.join(velocity_folder, subfolder)

        # if os.path.isdir(spectrogram_path) and os.path.isdir(range_path) and os.path.isdir(velocity_path):
        if os.path.isdir(spectrogram_path) and os.path.isdir(velocity_path):
            spectrogram_images = set([os.path.splitext(f)[0] for f in os.listdir(spectrogram_path)])
            # range_images = set([os.path.splitext(f.replace("_range", ""))[0] for f in os.listdir(range_path)])
            velocity_images = set([os.path.splitext(f.replace("_velocity", ""))[0] for f in os.listdir(velocity_path)])

            # missing_in_range = range_images - spectrogram_images
            # missing_in_range = spectrogram_images - range_images
            missing_in_velocity = velocity_images - spectrogram_images

            # print("Missing images in range:")
            # print(missing_in_range)

            print("\nMissing images in velocity:")
            print(missing_in_velocity)
            # for missing_image in missing_in_velocity:
            #     missing_image_path = os.path.join(velocity_path, f"{missing_image}_velocity.png")
            #     print("missing image path: ", missing_image_path)
            #     if os.path.exists(missing_image_path):
            #         os.remove(missing_image_path)
            #         print(f"Removed: {missing_image}_velocity.jpg")

find_missing_images()
