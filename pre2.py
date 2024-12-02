import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data_and_labels():
    images = []
    labels = []
    label_map = {}
    label_index = 0

    # Path to the IAM dataset folder
    image_folder = "D:/proj_5/iam_words/words"
    # Path to store processed data (renamed folder)
    output_path = "D:/proj_5/processed_data2"
    # Log file for failed images
    failed_images_log = "D:/proj_5/failed_images.txt"

    # Create the processed_data2 folder if it doesn't exist
    processed_data_folder = os.path.join(output_path, "processed_images")
    if not os.path.exists(processed_data_folder):
        os.makedirs(processed_data_folder)

    # Open the log file to store failed image paths
    with open(failed_images_log, 'w') as log_file:

        # Iterate through the top-level subdirectories (e.g., a01, a02, etc.)
        for subfolder in os.listdir(image_folder):
            subfolder_path = os.path.join(image_folder, subfolder)
            if os.path.isdir(subfolder_path):
                # Iterate through the nested subdirectories (e.g., a01-000u, a01-000x, etc.)
                for nested_subfolder in os.listdir(subfolder_path):
                    nested_subfolder_path = os.path.join(subfolder_path, nested_subfolder)
                    if os.path.isdir(nested_subfolder_path):
                        for image_file in os.listdir(nested_subfolder_path):
                            image_path = os.path.join(nested_subfolder_path, image_file)
                            # Only process image files
                            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):

                                print(f"Processing image: {image_path}")  # Debugging line
                                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                                if img is None:
                                    print(f"Failed to load image: {image_path}")  # Debugging line
                                    log_file.write(f"Failed to load image: {image_path}\n")  # Log failed image
                                    continue
                                # Resize to smaller size for better memory usage
                                img = cv2.resize(img, (128, 32))  # Resize to (128, 32) pixels for smaller memory footprint
                                img = img / 255.0  # Normalize pixel values
                                images.append(img)

                                # Create a label sequence for the image (e.g., mapping subfolder name to ASCII integers)
                                label = [ord(char) for char in subfolder]  # Convert each character in the subfolder name to ASCII
                                labels.append(label)

    # Pad the label sequences to ensure consistent length
    max_label_length = max(len(label) for label in labels)
    padded_labels = pad_sequences(labels, maxlen=max_label_length, padding='post', value=0)

    # Convert the images list to a numpy array
    images = np.array(images)

    # Save both images and labels in a single npz file in the processed_data2 folder
    np.savez_compressed(os.path.join(processed_data_folder, 'data.npz'), images=images, labels=padded_labels)

    print(f"Processed data saved to: {processed_data_folder}/data.npz")
    return images, padded_labels

# Main function to test loading of data
if __name__ == "__main__":
    images, labels = load_data_and_labels()
    print(f"Loaded {len(images)} images and {len(labels)} labels.")