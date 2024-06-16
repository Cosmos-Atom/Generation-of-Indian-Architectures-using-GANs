# mesa.py

from PIL import Image
from keras.preprocessing import image
import numpy as np

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

            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Additional debug information
            print("Image array created successfully.")

            # Make predictions using the classification model
            prediction = self.classification_model.predict(img_array)

            # Map numerical predictions to class labels
            predicted_class_index = np.argmax(prediction)
            predicted_class_label = self.get_class_label(predicted_class_index)
            predicted_probability = prediction[0, predicted_class_index]

            # Additional debug information
            print(f"Predicted Class: {predicted_class_label} | Probability: {predicted_probability}")

            # Return the result
            return f"Predicted Class: {predicted_class_label} | Probability: {predicted_probability}"

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
