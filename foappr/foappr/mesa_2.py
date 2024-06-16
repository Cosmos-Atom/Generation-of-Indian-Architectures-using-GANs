# mesa.py

from PIL import Image
from keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

class ImageProcessingAgent:
    def __init__(self, unique_id, model, classification_model):
        self.unique_id = unique_id
        self.model = model
        self.classification_model = classification_model
        self.class_labels = self.get_class_labels()

    def process_image(self, image_path):
        try:
            # Open the image using PIL (Python Imaging Library)
            img = Image.open(image_path)

            # Convert the image to RGB mode (3 channels)
            img = img.convert('RGB')

            # Additional debug information
            print("Image opened and converted to RGB successfully.")

            # Additional debug information
            print(f"Processing image: {image_path}")

            # Preprocess the image
            img = img.resize((128, 128))  # Resize to your target size

            # Additional debug information
            print("Image resized successfully.")

            
            cnn_model = self.classification_model

            # Load and preprocess the damaged image
            img = image.load_img(image_path, target_size=(128, 128))
            img_array = image.img_to_array(img)
            
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Make predictions
            predictions = cnn_model.predict(img_array)

            # Get the predicted class index
            predicted_class_index = np.argmax(predictions)

            # Get the class labels (folder names)
            class_labels = sorted(os.listdir('images\\train'))  # Assuming 'images\\train' is the path to your training data

            # Map the class index to the actual class label (folder name)
            predicted_monument = class_labels[predicted_class_index]


            # Additional debug information
            print(f"Predicted Class: {predicted_monument}")

            # Return the result
            return f"Predicted Class: {predicted_monument} "

        except Exception as e:
            # Print an exception message for debugging
            print(f"Exception during image processing: {e}")
            return f"Exception during image processing: {e}"

    def get_class_labels(self):
        # Retrieve class labels from the model (assuming the model has a 'class_names' attribute)
        return getattr(self.classification_model, 'class_names', [])

    def get_class_label(self, class_index):
        # Return the class label based on the provided class index
        return self.class_labels[class_index]
