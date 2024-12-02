import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.backend import ctc_batch_cost

def load_data_and_labels():
    images = []
    labels = []
    label_map = {}
    label_index = 0

    image_folder = "D:/proj_5/iam_words/words"
    output_path = "D:/proj_5/processed_data2"
    failed_images_log = "D:/proj_5/failed_images.txt"

    processed_data_folder = os.path.join(output_path, "processed_images")
    if not os.path.exists(processed_data_folder):
        os.makedirs(processed_data_folder)

    with open(failed_images_log, 'w') as log_file:

        for subfolder in os.listdir(image_folder):
            subfolder_path = os.path.join(image_folder, subfolder)
            if os.path.isdir(subfolder_path):
                for nested_subfolder in os.listdir(subfolder_path):
                    nested_subfolder_path = os.path.join(subfolder_path, nested_subfolder)
                    if os.path.isdir(nested_subfolder_path):
                        for image_file in os.listdir(nested_subfolder_path):
                            image_path = os.path.join(nested_subfolder_path, image_file)
                            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):

                                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                                if img is None:
                                    log_file.write(f"Failed to load image: {image_path}\n")
                                    continue
                                img = cv2.resize(img, (128, 32))
                                img = img / 255.0
                                images.append(img)

                                label = [ord(char) for char in subfolder]  
                                labels.append(label)

    max_label_length = max(len(label) for label in labels)
    padded_labels = pad_sequences(labels, maxlen=max_label_length, padding='post', value=0)

    images = np.array(images)

    np.savez_compressed(os.path.join(processed_data_folder, 'data.npz'), images=images, labels=padded_labels)

    return images, padded_labels

def ctc_loss(y_true, y_pred):
    return ctc_batch_cost(y_true, y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1], label_length=np.ones(y_true.shape[0]) * y_true.shape[1])

def build_crnn_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Reshape(target_shape=(-1, 64)))

    model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
    model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))

    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss=ctc_loss, metrics=['accuracy'])
    return model

def train_model():
    data = np.load("D:/proj_5/processed_data2/processed_images/data.npz")
    images = data['images']
    labels = data['labels']

    labels = labels.astype(str)

    unique_chars = sorted(set(''.join(labels.flatten())))
    char_to_index = {char: idx for idx, char in enumerate(unique_chars)}

    one_hot_labels = [np.array([char_to_index[char] for char in label]) for label in labels]

    train_images = images[:int(0.8 * len(images))]
    train_labels = one_hot_labels[:int(0.8 * len(labels))]
    val_images = images[int(0.8 * len(images)):]
    val_labels = one_hot_labels[int(0.8 * len(labels)):]

    train_images = train_images.reshape((-1, 32, 128, 1))  
    val_images = val_images.reshape((-1, 32, 128, 1))

    model = build_crnn_model((32, 128, 1), len(char_to_index))

    model.fit(train_images, np.array(train_labels), epochs=10, validation_data=(val_images, np.array(val_labels)), batch_size=32)

    model.save('htr_crnn_model.h5')
    print("Model saved as htr_crnn_model.h5")

if __name__ == "__main__":
    train_model()