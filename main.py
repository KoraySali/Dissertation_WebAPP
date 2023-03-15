from tensorflow.keras.models import load_model
from flask import *
import numpy as np
from PIL import Image
import time
from skimage import io, transform

app = Flask(__name__)

# Change this model to the complete model when done
test_model = load_model('C:/Users/koray/Downloads/Artefact/test_cnn.h5')


# Define the preprocess_images function
def preprocess_images(images, size=(124, 124, 3), padding=2):
    processed_images = []
    count = 0
    start_time = time.time()
    for image in images:
        img = image.astype(float)
        img = img.astype(float) / 255.0
        img = transform.resize(img, size)
        img = np.pad(img, ((padding, padding), (padding, padding), (0, 0)), 'constant')
        processed_images.append(img)
        count += 1
    end_time = time.time()
    print(f"Processed {count} images in {end_time - start_time:.2f} seconds.")
    return np.array(processed_images)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return "No file uploaded"
        img = io.imread(file.stream)
        img_array = preprocess_images([img])
        prediction = test_model.predict(img_array)
        # Format the prediction output
        formatted_prediction = ', '.join([f"{p:.2f}" for p in prediction[0]])
        return render_template('result.html', prediction=formatted_prediction)
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return "No file uploaded"
        try:
            img = Image.open(file)
        except:
            return "Unable to read the uploaded file"
        img = img.convert('RGB')  # Convert to RGB format if necessary
        img_array = preprocess_images([np.array(img)])
        prediction = test_model.predict(img_array)
        label = np.argmax(prediction, axis=-1)
        if label == 0:
            label_text = "Fire appears in the image."
        elif label == 1:
            label_text = "Smoke appears in the image."
        else:
            label_text = "No Fire or Smoke in the image."
    return render_template('result.html', prediction=label_text)


if __name__ == '__main__':
    app.run(debug=True)
    app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
