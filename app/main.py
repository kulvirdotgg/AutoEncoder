import torch
import io
import numpy as np
from PIL import Image
from typing import Union
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torchvision.transforms as transforms
from model.model import transform_img, add_noise, Autoencoder, Encoder, Decoder

app = Flask(__name__)
CORS(app)


@app.get('/')
def root():
    return {'message': 'Hello World!'}


@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    transformed_img = transform_img(image)
    model = Autoencoder()
    path = 'model/mnistae-0.1.0.pth'
    model.load_state_dict(torch.load(path))
    regenerated_img = model(transformed_img.unsqueeze(0)).squeeze()
    regenerated_img = regenerated_img.detach().numpy()
    regenerated_img = (regenerated_img * 255).astype('uint8')
    pil_image = Image.fromarray(regenerated_img, mode='L')
    image_stream = io.BytesIO()
    pil_image.save(image_stream, format='PNG')
    image_stream.seek(0)
    return send_file(image_stream, mimetype='image/png')


if __name__ == '__main__':
    app .run(port=8000, host='0.0.0.0', debug=True)
