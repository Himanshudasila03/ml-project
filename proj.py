import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
import numpy as np
import cv2

class HandwritingRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwriting Recognition")
        
        self.model = load_model('htr_crnn_model.h5')
        self.char_to_index = {str(i): i for i in range(10)}  # Adjust to match your actual character set
        self.index_to_char = {i: str(i) for i in range(10)}  # Adjust to match your actual character set
        
        self.label = tk.Label(self.root, text="Upload Image", font=("Helvetica", 14))
        self.label.pack(pady=20)

        self.upload_button = tk.Button(self.root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=20)

        self.result_label = tk.Label(self.root, text="Prediction: ", font=("Helvetica", 14))
        self.result_label.pack(pady=20)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 32))
            img = img / 255.0
            img = img.reshape((1, 32, 128, 1))

            prediction = self.model.predict(img)
            predicted_text = self.decode_ctc(prediction)
            self.result_label.config(text=f"Prediction: {predicted_text}")

    def decode_ctc(self, prediction):
        decoded = np.argmax(prediction, axis=-1)
        decoded = [self.index_to_char[idx] for idx in decoded[0] if idx != 0]
        return ''.join(decoded)

if __name__ == "__main__":
    root = tk.Tk()
    app = HandwritingRecognitionApp(root)
    root.mainloop()