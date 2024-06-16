# main.py

import os
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from mesa_2 import ImageProcessingAgent
import tensorflow as tf

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate')
def generate():
    return render_template('generate.html')

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


@app.route('/restore')
def restore():
    return render_template('restore.html')

if __name__ == '__main__':
    app.run(debug=True)
