# main.py
import torch
import os
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from mesa_2 import ImageProcessingAgent
import tensorflow as tf
from flask import render_template, send_from_directory, jsonify
from gen import generate_images, Generator, config
import config
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

import os

# Get the username
current_user = os.getenv('USER') or os.getenv('LOGNAME')

print(f"The Flask application is running under the user: {current_user}")

# Create the 'uploads' folder if it doesn't exist
uploads_folder = "uploads"
os.makedirs(uploads_folder, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for flash messages

# Load the image classifier model
model_path = "indian_monuments_recognition_model2_vgg16.h5"
classification_model = tf.keras.models.load_model(model_path)

# Initialize your MESA agent with the loaded model
your_mesa_agent = ImageProcessingAgent(unique_id=1, model=None, classification_model=classification_model)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Load the generator model
Z_DIM = 256
IN_CHANNELS = 256
gen = Generator(Z_DIM, IN_CHANNELS, img_channels=3).to(config.DEVICE)

# Load the generator model checkpoint
checkpoint_file = "generator.pth"
checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
gen.load_state_dict(checkpoint["state_dict"])

# Set the steps
steps = 5


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate')
def generate():
    # Create the directory if it doesn't exist
    save_folder = "saved_images"
    os.makedirs(save_folder, exist_ok=True)

    # Generate images
    generate_images(gen, steps)

    # Return JSON response
    response = jsonify({'status': 'success'})

    # Render generate.html template
    return render_template('generate.html', response=response)

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)

        image_file = request.files['image']

        if image_file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        try:
            if image_file and allowed_file(image_file.filename):
                # Save the image to the "uploads" folder
                filename = secure_filename(image_file.filename)
                file_path = os.path.join(uploads_folder, filename)
                image_file.save(file_path)

                # Pass the image to your MESA agent for processing
                result = your_mesa_agent.process_image(file_path)

                # Do something with the result (e.g., display it on a new page)
                return render_template('result.html', result=result)

            flash('Invalid file format. Please upload an image.', 'error')

        except Exception as e:
            # Print an exception message for debugging
            print(f"Exception during file upload: {e}")

    return render_template('classify.html')


@app.route('/display_images')
def display_images():
    # Get the list of generated image paths
    generated_image_paths = ['saved_images/' + filename for filename in os.listdir('static/saved_images')]

    return render_template('display_images.html', image_paths=generated_image_paths)

# Define the function to load and predict
def load_and_predict(model_path, image_path):
    # Load the trained CNN model
    cnn_model = load_model(model_path)

    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make predictions
    predictions = cnn_model.predict(img_array)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions)

    # Get the class labels (folder names)
    class_labels = sorted(os.listdir('images/train'))

    # Map the class index to the actual class label (folder name)
    predicted_monument = class_labels[predicted_class_index]

    return predicted_monument

@app.route('/restore', methods=['GET', 'POST'])
def restore():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)

        image_file = request.files['image']

        if image_file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        
        
        # Save the image to the "uploads" folder
        filename = secure_filename(image_file.filename)
        file_path = os.path.join(uploads_folder, filename)
        image_file.save(file_path)

        # Make predictions on the restored image
        predicted_monument = load_and_predict(model_path, file_path)

        # Pass the predicted monument name to the template
        return render_template('restored_results.html', predicted_monument=predicted_monument)

        flash('Invalid file format. Please upload an image.', 'error')


    return render_template('restore.html')



if __name__ == '__main__':
    app.run(debug=True)
